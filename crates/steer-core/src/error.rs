//! Error types for steering operations.

use thiserror::Error;

/// The error type returned by all `steer-core` operations.
#[derive(Debug, Error)]
pub enum SteerError {
    #[error("invalid layer {layer} (max {max})")]
    InvalidLayer { layer: usize, max: usize },
    #[error("invalid neuron {neuron} at layer {layer}")]
    InvalidNeuron { layer: usize, neuron: usize },
    #[error("invalid site kind: {0}")]
    InvalidSiteKind(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("invalid mode: {0}")]
    InvalidMode(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("empty intervention set")]
    EmptyInterventionSet,
}

/// Short alias used throughout the crate.
pub type Result<T> = std::result::Result<T, SteerError>;
