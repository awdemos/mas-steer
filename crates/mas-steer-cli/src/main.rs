//! Command-line interface for MAS-Steer.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use agent::{generate_response, resume_search, run_search, Checkpoint, SearchConfig};
use steer_core::load_model;

#[cfg(feature = "mock")]
use steer_core::{MockBackend, ModelDims};

const MAX_PROMPT_FILE_BYTES: u64 = 1_048_576;

/// MAS-Steer: search for steering interventions.
#[derive(Parser, Debug)]
#[command(name = "mas-steer")]
#[command(about = "Multi-Agent Search over neuron intervention spaces")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a search against a real model backend.
    Search {
        /// Model directory or Hugging Face repo ID (e.g. "Qwen/Qwen2.5-0.5B").
        #[arg(short, long)]
        model: String,
        /// Path to a TOML search-config file.
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Number of search generations.
        #[arg(short, long, default_value_t = 10)]
        generations: usize,
        /// Population size.
        #[arg(short, long, default_value_t = 20)]
        population: usize,
        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Number of task prompts to generate.
        #[arg(long, default_value_t = 12)]
        tasks: usize,
        /// Number of control prompts.
        #[arg(long, default_value_t = 6)]
        controls: usize,
        /// Output JSON path for the champion.
        #[arg(short, long, default_value = "champion.json")]
        output: PathBuf,
        /// Directory to write per-generation checkpoint files.
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        /// Write a checkpoint every N generations.
        #[arg(long, default_value_t = 1)]
        checkpoint_every: usize,
    },
    /// Resume a search from a checkpoint JSON.
    Resume {
        /// Model directory or Hugging Face repo ID.
        #[arg(short, long)]
        model: String,
        /// Path to a checkpoint file produced by search.
        #[arg(short, long)]
        checkpoint: PathBuf,
        /// Additional generations to run.
        #[arg(short, long, default_value_t = 10)]
        generations: usize,
        /// Output JSON path for the champion.
        #[arg(short, long, default_value = "champion.json")]
        output: PathBuf,
        /// Directory to write per-generation checkpoint files.
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        /// Write a checkpoint every N generations.
        #[arg(long, default_value_t = 1)]
        checkpoint_every: usize,
    },
    /// Evaluate a saved champion against a set of prompts.
    Eval {
        /// Model directory or Hugging Face repo ID.
        #[arg(short, long)]
        model: String,
        /// Path to the champion JSON file.
        #[arg(short, long)]
        champion: PathBuf,
        /// Path to a newline-separated prompt file.
        #[arg(short, long)]
        prompts: PathBuf,
    },
    /// Download a model from Hugging Face Hub to a local directory.
    Pull {
        /// Hugging Face repo ID.
        repo_id: String,
        /// Local directory to stage the model files.
        #[arg(short, long, default_value = "./models")]
        output_dir: PathBuf,
    },
}

fn load_backend(model: &str) -> Result<Arc<dyn steer_core::Backend>> {
    #[cfg(feature = "mock")]
    if model == "mock" {
        return Ok(Arc::new(MockBackend::new(
            ModelDims {
                layers: 6,
                hidden_dim: 32,
                mlp_dim: 64,
                vocab_size: 128,
            },
            42,
        )));
    }
    Ok(Arc::new(load_model(model)?))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt::init();

    match cli.command {
        Commands::Search {
            model,
            config,
            generations,
            population,
            seed,
            tasks,
            controls,
            output,
            checkpoint_dir,
            checkpoint_every,
        } => {
            let mut search_config = match config {
                Some(p) => load_search_config(&p)?,
                None => SearchConfig::default(),
            };
            search_config.generations = generations;
            search_config.population_size = population;
            search_config.checkpoint_dir = checkpoint_dir.map(|p| p.to_string_lossy().to_string());
            search_config.checkpoint_every = checkpoint_every;

            let backend: Arc<dyn steer_core::Backend> = load_backend(&model)?;

            info!(
                "starting search: {} generations, population {}",
                search_config.generations, search_config.population_size
            );
            let result = run_search(backend, search_config, seed, tasks, controls)
                .context("search failed")?;

            write_result(&output, &result)?;
        }
        Commands::Resume {
            model,
            checkpoint,
            generations,
            output,
            checkpoint_dir,
            checkpoint_every,
        } => {
            let (mut search_config, checkpoint_data) = load_checkpoint(&checkpoint)?;
            search_config.generations = generations;
            search_config.checkpoint_dir = checkpoint_dir.map(|p| p.to_string_lossy().to_string());
            search_config.checkpoint_every = checkpoint_every;

            let backend: Arc<dyn steer_core::Backend> = load_backend(&model)?;
            let seed = seed_from_path(&checkpoint);

            info!(
                "resuming search from generation {} for {} more generations",
                checkpoint_data.generation, generations
            );
            let result = resume_search(backend, search_config, checkpoint_data, seed)
                .context("resume failed")?;

            write_result(&output, &result)?;
        }
        Commands::Eval {
            model,
            champion,
            prompts,
        } => {
            let candidate = load_champion(&champion)?;
            let prompt_text = read_limited_text(&prompts, MAX_PROMPT_FILE_BYTES)
                .with_context(|| format!("reading prompts {}", prompts.display()))?;
            let backend: Arc<dyn steer_core::Backend> = load_backend(&model)?;

            for line in prompt_text.lines().filter(|l| !l.trim().is_empty()) {
                let response = generate_response(backend.as_ref(), line, &candidate.interventions);
                println!("PROMPT: {line}\nRESPONSE: {response}\n");
            }
        }
        Commands::Pull {
            repo_id,
            output_dir,
        } => {
            let client = hf_hub::HFClientSync::new().context("cannot create HF hub client")?;
            let owner_name: Vec<&str> = repo_id.splitn(2, '/').collect();
            let (owner, name) = if owner_name.len() == 2 {
                (owner_name[0], owner_name[1])
            } else {
                ("", repo_id.as_str())
            };
            let repo = client.model(owner, name);
            let dir = output_dir.join(repo_id.replace('/', "--"));
            fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            let paths = [
                ("config.json", dir.join("config.json")),
                ("model.safetensors", dir.join("model.safetensors")),
                ("tokenizer.json", dir.join("tokenizer.json")),
            ];
            for (remote, local) in &paths {
                let fetched = repo
                    .download_file()
                    .filename(remote.to_string())
                    .send()
                    .with_context(|| format!("fetching {remote} from {repo_id}"))?;
                fs::copy(&fetched, local)
                    .with_context(|| format!("copying to {}", local.display()))?;
                info!("{} -> {}", remote, local.display());
            }
            println!("Model staged at {}", dir.display());
        }
    }

    Ok(())
}

fn write_result(output: &Path, result: &agent::SearchResult) -> Result<()> {
    let json = serde_json::to_string_pretty(result).context("failed to serialize search result")?;
    validate_output_path(output)?;
    fs::write(output, json).with_context(|| format!("writing {}", output.display()))?;
    info!(
        "champion fitness {} written to {}",
        result.champion.fitness,
        output.display()
    );
    Ok(())
}

fn load_search_config(path: &Path) -> Result<SearchConfig> {
    let text =
        fs::read_to_string(path).with_context(|| format!("reading config {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
}

fn load_champion(path: &Path) -> Result<agent::Candidate> {
    let text =
        fs::read_to_string(path).with_context(|| format!("reading champion {}", path.display()))?;
    let result: agent::SearchResult = serde_json::from_str(&text)
        .with_context(|| format!("parsing champion {}", path.display()))?;
    Ok(result.champion.candidate)
}

/// Reject output paths that traverse parent directories.
fn validate_output_path(path: &Path) -> Result<()> {
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            bail!(
                "refusing path with parent-directory traversal: {}",
                path.display()
            );
        }
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            bail!("output directory does not exist: {}", parent.display());
        }
    }
    Ok(())
}

/// Read a text file, failing if it exceeds `max_bytes`.
fn read_limited_text(path: &Path, max_bytes: u64) -> Result<String> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let len = file.metadata().map(|m| m.len()).unwrap_or(max_bytes + 1);
    if len > max_bytes {
        bail!("prompt file too large: {} bytes (max {})", len, max_bytes);
    }
    let mut buf = Vec::with_capacity(len as usize);
    let mut reader = file.take(max_bytes);
    reader
        .read_to_end(&mut buf)
        .with_context(|| format!("reading {}", path.display()))?;
    if buf.len() > max_bytes as usize {
        bail!("prompt file exceeded size limit of {} bytes", max_bytes);
    }
    String::from_utf8(buf).with_context(|| format!("{} is not valid UTF-8", path.display()))
}

fn seed_from_path(path: &Path) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    h.finish()
}

fn load_checkpoint(path: &Path) -> Result<(SearchConfig, Checkpoint)> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading checkpoint {}", path.display()))?;
    let cp: Checkpoint = serde_json::from_str(&text)
        .with_context(|| format!("parsing checkpoint {}", path.display()))?;
    Ok((SearchConfig::default(), cp))
}
