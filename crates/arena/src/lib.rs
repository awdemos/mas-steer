#![allow(missing_docs)]

pub mod challenger;
pub mod scoring;
pub mod task;

pub use challenger::Challenger;
pub use scoring::{control_penalty, total_strength, FitnessConfig, Score};
pub use task::{TaskFamily, TaskInstance};
