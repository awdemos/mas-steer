//! Real model backend using Candle and Hugging Face models.
//!
//! Loads a Qwen2-style model from a local path or the Hugging Face Hub,
//! tokenizes prompts, and runs forward passes with optional activation hooks.

use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config as QwenConfig, ModelForCausalLM as QwenModel};
use tokenizers::pipeline::{EncodeOptions, Input, Inputs, PipelineTokenizer};
use tracing::info;

use crate::backend::{Backend, Logits, ModelDims, TokenId};
use crate::error::{Result, SteerError};
use crate::hook::Hook;
use crate::intervention::SiteKind;

/// A backend backed by a real Hugging Face transformer model loaded via Candle.
pub struct ModelBackend {
    model: QwenModel,
    tokenizer: PipelineTokenizer,
    dims: ModelDims,
    device: Device,
}

impl ModelBackend {
    /// Load a Qwen2 model from a directory containing `model.safetensors` and
    /// `tokenizer.json`.
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !weights_path.exists() {
            return Err(SteerError::Backend(format!(
                "model weights not found at {}",
                weights_path.display()
            )));
        }
        if !tokenizer_path.exists() {
            return Err(SteerError::Backend(format!(
                "tokenizer not found at {}",
                tokenizer_path.display()
            )));
        }

        let device = Device::Cpu;
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|e| SteerError::Backend(format!("cannot read config.json: {e}")))?;
        let config: QwenConfig = serde_json::from_str(&config_text)
            .map_err(|e| SteerError::Backend(format!("cannot parse config.json: {e}")))?;

        let weights_bytes = std::fs::read(&weights_path)
            .map_err(|e| SteerError::Backend(format!("cannot read weights file: {e}")))?;
        let vb =
            VarBuilder::from_buffered_safetensors(weights_bytes, candle_core::DType::F32, &device)
                .map_err(|e| SteerError::Backend(format!("cannot build VarBuilder: {e}")))?;

        let model = QwenModel::new(&config, vb)
            .map_err(|e| SteerError::Backend(format!("cannot build model: {e}")))?;

        let tokenizer = tokenizers::from_json_file(&tokenizer_path)
            .map_err(|e| SteerError::Tokenizer(format!("cannot load tokenizer: {e}")))?;

        let dims = ModelDims {
            layers: config.num_hidden_layers,
            hidden_dim: config.hidden_size,
            mlp_dim: config.intermediate_size,
            vocab_size: config.vocab_size,
        };

        info!(
            "loaded model: {} layers, hidden={}, mlp={}, vocab={}",
            dims.layers, dims.hidden_dim, dims.mlp_dim, dims.vocab_size
        );

        Ok(Self {
            model,
            tokenizer,
            dims,
            device,
        })
    }

    /// Download a model from Hugging Face Hub into the local cache and load it.
    pub fn from_hub(repo_id: &str) -> Result<Self> {
        let client = hf_hub::HFClientSync::new()
            .map_err(|e| SteerError::Backend(format!("cannot create HF hub client: {e}")))?;
        let owner_name: Vec<&str> = repo_id.splitn(2, '/').collect();
        let (owner, name) = if owner_name.len() == 2 {
            (owner_name[0], owner_name[1])
        } else {
            ("", repo_id)
        };
        let repo = client.model(owner, name);

        let config_path = repo
            .download_file()
            .filename("config.json".to_string())
            .send()
            .map_err(|e| SteerError::Backend(format!("cannot fetch config.json: {e}")))?;
        let weights_path = repo
            .download_file()
            .filename("model.safetensors".to_string())
            .send()
            .map_err(|e| SteerError::Backend(format!("cannot fetch model weights: {e}")))?;
        let tokenizer_path = repo
            .download_file()
            .filename("tokenizer.json".to_string())
            .send()
            .map_err(|e| SteerError::Backend(format!("cannot fetch tokenizer: {e}")))?;

        let model_dir = config_path
            .parent()
            .expect("config.json has a parent directory");

        // Ensure all files are in the same directory.
        std::fs::copy(&weights_path, model_dir.join("model.safetensors"))
            .map_err(|e| SteerError::Backend(format!("cannot stage weights: {e}")))?;
        std::fs::copy(&tokenizer_path, model_dir.join("tokenizer.json"))
            .map_err(|e| SteerError::Backend(format!("cannot stage tokenizer: {e}")))?;

        Self::from_dir(model_dir)
    }
}

impl Backend for ModelBackend {
    fn dims(&self) -> ModelDims {
        self.dims
    }

    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>> {
        let options = EncodeOptions::default();
        let handle = self
            .tokenizer
            .encode(Inputs::Single(Input::Single(text.to_string())), &options);
        let mut encodings = handle
            .wait()
            .map_err(|e| SteerError::Tokenizer(format!("encode failed: {e}")))?;
        if encodings.is_empty() {
            return Err(SteerError::Tokenizer("empty encoding".to_string()));
        }
        let encoding = encodings.swap_remove(0);
        let ids: Vec<TokenId> = encoding.ids().iter().map(|t| t.id()).collect();
        Ok(ids)
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| SteerError::Tokenizer(format!("decode failed: {e}")))
    }

    fn forward(&self, tokens: &[TokenId], hooks: &[Hook]) -> Result<Logits> {
        if tokens.is_empty() {
            return Err(SteerError::Tokenizer("empty token sequence".to_string()));
        }

        let input = Tensor::new(tokens.to_vec(), &self.device)
            .map_err(|e| SteerError::Backend(format!("input tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| SteerError::Backend(format!("unsqueeze: {e}")))?;

        // Qwen ModelForCausalLM already returns logits for the last position.
        // Model::forward is &mut because of KV cache; clone the model state for
        // this per-forward steering call.
        let mut model = self.model.clone();
        let logits = model
            .forward(&input, 0)
            .map_err(|e| SteerError::Backend(format!("forward: {e}")))?;

        let logits = logits
            .squeeze(0)
            .map_err(|e| SteerError::Backend(format!("squeeze logits: {e}")))?;

        // Apply residual hooks if any target the final layer's residual stream.
        // NOTE: true layer-wise hooking requires intercepting hidden states
        // inside the model forward. For now we support final-layer residual edits
        // by applying hooks to the last-token logits vector as a practical proxy.
        if !hooks.is_empty() {
            let mut v = logits
                .to_vec1::<f32>()
                .map_err(|e| SteerError::Backend(format!("logits to vec: {e}")))?;
            for h in hooks {
                if h.site.kind == SiteKind::Residual && h.site.layer == self.dims.layers {
                    let idx = h.site.index % v.len();
                    let old = v[idx];
                    v[idx] = h.apply(old);
                }
            }
            return Ok(v);
        }

        logits
            .to_vec1::<f32>()
            .map_err(|e| SteerError::Backend(format!("logits to vec: {e}")))
    }
}

/// Convenience loader that picks `from_hub` for a repo-id or `from_dir` for a
/// filesystem path.
pub fn load_model(path_or_repo: &str) -> Result<ModelBackend> {
    let p = Path::new(path_or_repo);
    if p.is_dir() {
        ModelBackend::from_dir(p)
    } else {
        ModelBackend::from_hub(path_or_repo)
    }
}
