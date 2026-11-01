//! Runtime hooks applied during a forward pass.

use crate::intervention::{Mode, Site};

/// A runtime intervention ready to be applied to an activation value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hook {
    pub site: Site,
    pub mode: Mode,
    pub dose: f32,
}

impl Hook {
    /// Apply this hook to a scalar activation.
    #[inline]
    #[must_use]
    pub fn apply(&self, value: f32) -> f32 {
        self.mode.apply(value, self.dose)
    }
}
