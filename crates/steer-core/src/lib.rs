#![allow(missing_docs)]

pub mod backend;
pub mod error;
pub mod hook;
pub mod intervention;

#[cfg(feature = "mock")]
pub mod mock_backend;

#[cfg(feature = "model")]
pub mod model_backend;

pub use backend::{Backend, Logits, ModelDims, TokenId};
pub use error::{Result, SteerError};
pub use hook::Hook;
pub use intervention::{Intervention, Mode, Site, SiteKind};

#[cfg(feature = "mock")]
pub use mock_backend::MockBackend;

#[cfg(feature = "model")]
pub use model_backend::{load_model, ModelBackend};
