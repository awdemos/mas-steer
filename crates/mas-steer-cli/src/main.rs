//! Command-line interface for MAS-Steer.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use tracing::info;

use agent::{run_search, SearchConfig};
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
    /// Run a search against a mock or real backend.
    Search {
        /// Number of transformer layers.
        #[arg(long, default_value_t = 6)]
        layers: usize,
        /// Hidden dimension size.
        #[arg(long, default_value_t = 32)]
        hidden_dim: usize,
        /// MLP intermediate dimension.
        #[arg(long, default_value_t = 64)]
        mlp_dim: usize,
        /// Vocabulary size.
        #[arg(long, default_value_t = 128)]
        vocab_size: usize,
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
    },
    /// Evaluate a saved champion against a set of prompts.
    Eval {
        /// Path to the champion JSON file.
        #[arg(short, long)]
        champion: PathBuf,
        /// Path to a newline-separated prompt file.
        #[arg(short, long)]
        prompts: PathBuf,
        /// Mock backend layers.
        #[arg(long, default_value_t = 6)]
        layers: usize,
        /// Mock backend hidden dim.
        #[arg(long, default_value_t = 32)]
        hidden_dim: usize,
        /// Mock backend MLP dim.
        #[arg(long, default_value_t = 64)]
        mlp_dim: usize,
        /// Mock backend vocab size.
        #[arg(long, default_value_t = 128)]
        vocab_size: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt::init();

    match cli.command {
        Commands::Search {
            layers,
            hidden_dim,
            mlp_dim,
            vocab_size,
            config,
            generations,
            population,
            seed,
            tasks,
            controls,
            output,
        } => {
            let mut search_config = match config {
                Some(p) => load_search_config(&p)?,
                None => SearchConfig::default(),
            };
            search_config.generations = generations;
            search_config.population_size = population;

            let dims = ModelDims {
                layers,
                hidden_dim,
                mlp_dim,
                vocab_size,
            };
            let backend: Arc<dyn steer_core::Backend> = Arc::new(MockBackend::new(dims, seed));

            info!(
                "starting search: {} generations, population {}",
                search_config.generations, search_config.population_size
            );
            let result = run_search(backend, search_config, seed, tasks, controls)
                .context("search failed")?;

            let json = serde_json::to_string_pretty(&result)
                .context("failed to serialize search result")?;
            validate_output_path(&output)?;
            fs::write(&output, json).with_context(|| format!("writing {}", output.display()))?;
            info!(
                "champion fitness {} written to {}",
                result.champion.fitness,
                output.display()
            );
        }
        Commands::Eval {
            champion,
            prompts,
            layers,
            hidden_dim,
            mlp_dim,
            vocab_size,
        } => {
            let candidate = load_champion(&champion)?;
            let prompt_text = read_limited_text(&prompts, MAX_PROMPT_FILE_BYTES)
                .with_context(|| format!("reading prompts {}", prompts.display()))?;
            let dims = ModelDims {
                layers,
                hidden_dim,
                mlp_dim,
                vocab_size,
            };
            let backend: Arc<dyn steer_core::Backend> =
                Arc::new(MockBackend::new(dims, seed_from_path(&champion)));

            for line in prompt_text.lines().filter(|l| !l.trim().is_empty()) {
                let response = agent::search::generate_response(
                    backend.as_ref(),
                    line,
                    &candidate.interventions,
                );
                println!(
                    "PROMPT: {line}
RESPONSE: {response}
"
                );
            }
        }
    }

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
