//! Deterministic mock backend for tests and algorithmic benchmarks.

use std::collections::HashMap;

use rand::distributions::{Distribution, Standard};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::backend::{Backend, Logits, ModelDims, TokenId};
use crate::error::{Result, SteerError};
use crate::hook::Hook;
use crate::intervention::SiteKind;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// A deterministic, lightweight transformer-ish model for testing steering algorithms.
pub struct MockBackend {
    dims: ModelDims,
    vocab: HashMap<String, TokenId>,
    reverse_vocab: HashMap<TokenId, String>,
    unk: TokenId,
    up: Vec<Vec<f32>>,   // layers x hidden x mlp
    down: Vec<Vec<f32>>, // layers x mlp x hidden
    out: Vec<f32>,       // hidden x vocab
    rng_seed: u64,
}

impl MockBackend {
    /// Build a mock model with the given dimensions and a fixed seed.
    pub fn new(dims: ModelDims, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut vocab = HashMap::new();
        let mut reverse = HashMap::new();
        // Reserve 0 for padding/unknown, 1..vocab-1 for known tokens.
        vocab.insert("<pad>".to_string(), 0);
        reverse.insert(0, "<pad>".to_string());

        let words = [
            "the", "a", "an", "story", "happy", "sad", "tomato", "evil", "villain", "hero", "cat",
            "dog", "ran", "sat", "jumped", "laughed", "cried", "dark", "bright", "yes", "no", "is",
            "was", "will", "and", "or", "not", "very", "slightly",
        ];
        for (i, w) in words.iter().enumerate() {
            let id = (i + 1) as TokenId;
            if (id as usize) < dims.vocab_size {
                vocab.insert(w.to_string(), id);
                reverse.insert(id, w.to_string());
            }
        }

        let up = Self::layered_mat(&mut rng, dims.layers, dims.hidden_dim, dims.mlp_dim);
        let down = Self::layered_mat(&mut rng, dims.layers, dims.mlp_dim, dims.hidden_dim);
        let out = Self::vec(&mut rng, dims.hidden_dim * dims.vocab_size);

        let unk = 0;

        Self {
            dims,
            vocab,
            reverse_vocab: reverse,
            unk,
            up,
            down,
            out,
            rng_seed: seed,
        }
    }

    fn vec(rng: &mut ChaCha8Rng, len: usize) -> Vec<f32> {
        Standard
            .sample_iter(rng)
            .take(len)
            .map(|x: f32| x - 0.5)
            .collect()
    }

    fn layered_mat(rng: &mut ChaCha8Rng, layers: usize, rows: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..layers).map(|_| Self::vec(rng, rows * cols)).collect()
    }

    fn matvec(mat: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
        debug_assert_eq!(vec.len(), cols);
        (0..rows)
            .map(|r| {
                let mut acc = 0.0_f32;
                for c in 0..cols {
                    acc += mat[r * cols + c] * vec[c];
                }
                acc
            })
            .collect()
    }

    fn hidden_for_token(&self, id: TokenId) -> Vec<f32> {
        // Deterministic embedding: hash the token id into a unit-ish vector.
        let dim = self.dims.hidden_dim;
        let mut v = vec![0.0_f32; dim];
        let mut rng = ChaCha8Rng::seed_from_u64(self.rng_seed.wrapping_add(u64::from(id)));
        for x in &mut v {
            *x = rng.gen::<f32>() - 0.5;
        }
        v
    }

    fn apply_residual_hooks(&self, hidden: &mut [f32], hooks: &[Hook], layer: usize) {
        for h in hooks {
            if h.site.kind == SiteKind::Residual && h.site.layer == layer {
                hidden[h.site.index] = h.apply(hidden[h.site.index]);
            }
        }
    }

    fn apply_mlp_hooks(&self, activations: &mut [f32], hooks: &[Hook], layer: usize) {
        for h in hooks {
            if h.site.kind == SiteKind::Mlp && h.site.layer == layer {
                activations[h.site.index] = h.apply(activations[h.site.index]);
            }
        }
    }
}

impl Backend for MockBackend {
    fn dims(&self) -> ModelDims {
        self.dims
    }

    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        for word in text.split_whitespace() {
            let cleaned: String = word
                .to_ascii_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect();
            let id = self.vocab.get(&cleaned).copied().unwrap_or(self.unk);
            out.push(id);
        }
        if out.is_empty() {
            out.push(self.unk);
        }
        Ok(out)
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        let words: Vec<String> = tokens
            .iter()
            .map(|t| {
                self.reverse_vocab
                    .get(t)
                    .cloned()
                    .unwrap_or_else(|| "<unk>".to_string())
            })
            .collect();
        Ok(words.join(" "))
    }

    fn forward(&self, tokens: &[TokenId], hooks: &[Hook]) -> Result<Logits> {
        if tokens.is_empty() {
            return Err(SteerError::Tokenizer("empty token sequence".to_string()));
        }

        let dims = self.dims;
        // Compute hidden state for the last token, applying hooks per layer.
        let mut hidden = self.hidden_for_token(tokens[0]);
        for (layer, (up_layer, down_layer)) in self.up.iter().zip(self.down.iter()).enumerate() {
            let mlp_pre = Self::matvec(up_layer, dims.mlp_dim, dims.hidden_dim, &hidden);
            let mut mlp_post: Vec<f32> = mlp_pre.into_iter().map(relu).collect();
            self.apply_mlp_hooks(&mut mlp_post, hooks, layer);
            let update = Self::matvec(down_layer, dims.hidden_dim, dims.mlp_dim, &mlp_post);
            for (h, u) in hidden.iter_mut().zip(update) {
                *h += u;
            }
            self.apply_residual_hooks(&mut hidden, hooks, layer);
        }

        let logits = Self::matvec(&self.out, dims.vocab_size, dims.hidden_dim, &hidden);
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> MockBackend {
        MockBackend::new(
            ModelDims {
                layers: 4,
                hidden_dim: 16,
                mlp_dim: 32,
                vocab_size: 64,
            },
            42,
        )
    }

    #[test]
    fn roundtrip_tokenize_decode() {
        let b = backend();
        let text = "the happy cat sat";
        let ids = b.tokenize(text).unwrap();
        let back = b.decode(&ids).unwrap();
        assert_eq!(back, text);
    }

    #[test]
    fn forward_changes_with_hooks() {
        let b = backend();
        let ids = b.tokenize("happy").unwrap();
        let base = b.forward(&ids, &[]).unwrap();
        let hook = Hook {
            site: crate::intervention::Site {
                kind: SiteKind::Residual,
                layer: 0,
                index: 0,
            },
            mode: crate::intervention::Mode::Add,
            dose: 10.0,
        };
        let steered = b.forward(&ids, &[hook]).unwrap();
        assert_ne!(base, steered);
    }
}
