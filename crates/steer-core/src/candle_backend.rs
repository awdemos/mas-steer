//! Optional Candle backend integration.
//!
//! This module is gated behind the `candle` feature. It provides a thin
//! trait bridge so real Candle models can be wrapped as `Backend` implementations
//! while keeping the default build fast.

use crate::backend::{Backend, Logits, ModelDims, TokenId};
use crate::error::Result;
use crate::hook::Hook;

type TokenizeFn = Box<dyn Fn(&str) -> Result<Vec<TokenId>> + Send + Sync>;
type DecodeFn = Box<dyn Fn(&[TokenId]) -> Result<String> + Send + Sync>;
type ForwardFn = Box<dyn Fn(&[TokenId], &[Hook]) -> Result<Logits> + Send + Sync>;

/// A backend driven by a user-provided Candle forward closure.
///
/// The closure receives the input tensor and any hooks; it is responsible for
/// running the model and returning the final hidden vector.
pub struct CandleBackend {
    dims: ModelDims,
    tokenize_fn: TokenizeFn,
    decode_fn: DecodeFn,
    forward_fn: ForwardFn,
}

impl CandleBackend {
    /// Construct a Candle-backed backend from closures.
    pub fn new(
        dims: ModelDims,
        tokenize: impl Fn(&str) -> Result<Vec<TokenId>> + Send + Sync + 'static,
        decode: impl Fn(&[TokenId]) -> Result<String> + Send + Sync + 'static,
        forward: impl Fn(&[TokenId], &[Hook]) -> Result<Logits> + Send + Sync + 'static,
    ) -> Self {
        Self {
            dims,
            tokenize_fn: Box::new(tokenize),
            decode_fn: Box::new(decode),
            forward_fn: Box::new(forward),
        }
    }
}

impl Backend for CandleBackend {
    fn dims(&self) -> ModelDims {
        self.dims
    }

    fn forward(&self, tokens: &[TokenId], hooks: &[Hook]) -> Result<Logits> {
        (self.forward_fn)(tokens, hooks)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>> {
        (self.tokenize_fn)(text)
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        (self.decode_fn)(tokens)
    }
}

/// Re-export candle core so downstream code can build tensors without adding
/// the dependency directly.
#[cfg(feature = "candle")]
pub use candle_core;
