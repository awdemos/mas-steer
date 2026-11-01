//! Abstract model backend trait.

use crate::error::Result;
use crate::hook::Hook;

/// Token id used by all backends.
pub type TokenId = u32;
/// Logits over the vocabulary for the final token.
pub type Logits = Vec<f32>;

/// Dimensions of a transformer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDims {
    pub layers: usize,
    pub hidden_dim: usize,
    pub mlp_dim: usize,
    pub vocab_size: usize,
}

/// A model that can run a steered forward pass.
pub trait Backend: Send + Sync {
    /// Return the model dimensions.
    fn dims(&self) -> ModelDims;

    /// Run a forward pass, applying hooks where requested.
    fn forward(&self, tokens: &[TokenId], hooks: &[Hook]) -> Result<Logits>;

    /// Tokenize a text prompt.
    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>>;

    /// Decode token ids back to text.
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
}
