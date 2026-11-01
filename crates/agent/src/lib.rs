#![allow(missing_docs)]

pub mod mutation;
pub mod population;
pub mod search;

pub use population::{Candidate, Champion};
pub use search::{evaluate_candidate, run_search, SearchConfig, SearchResult};
