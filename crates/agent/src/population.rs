//! Candidate representation and archive tracking.

use serde::{Deserialize, Serialize};
use steer_core::Intervention;

/// A candidate set of interventions.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub interventions: Vec<Intervention>,
}

impl Candidate {
    /// Create an empty candidate.
    pub fn empty() -> Self {
        Self {
            interventions: Vec::new(),
        }
    }

    /// Return a new candidate with duplicate sites removed, keeping the first.
    #[must_use]
    pub fn deduped(mut self) -> Self {
        let mut seen = std::collections::HashSet::new();
        self.interventions
            .retain(|i| seen.insert((i.site.kind, i.site.layer, i.site.index)));
        self
    }
}

/// A scored, archived champion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Champion {
    pub candidate: Candidate,
    pub generation: usize,
    pub fitness: f32,
}
