//! Structured, serializable intervention definitions.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::error::{Result, SteerError};
use crate::hook::Hook;

/// How an intervention modifies a target value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Add a signed offset.
    Add,
    /// Multiply by `(1 + dose)`.
    Scale,
    /// Clamp the absolute value to `|dose|` in the original direction.
    Clamp,
    /// Force the target to zero.
    Zero,
}

impl Mode {
    /// Apply this mode to a single scalar.
    #[must_use]
    pub fn apply(self, value: f32, dose: f32) -> f32 {
        match self {
            Mode::Add => value + dose,
            Mode::Scale => value * (1.0 + dose),
            Mode::Clamp => {
                let sign = value.signum();
                let bound = dose.abs();
                if value.abs() > bound {
                    sign * bound
                } else {
                    value
                }
            }
            Mode::Zero => 0.0,
        }
    }
}

impl FromStr for Mode {
    type Err = SteerError;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "add" => Ok(Mode::Add),
            "scale" => Ok(Mode::Scale),
            "clamp" => Ok(Mode::Clamp),
            "zero" => Ok(Mode::Zero),
            _ => Err(SteerError::InvalidMode(s.to_string())),
        }
    }
}

/// The kind of site an intervention targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SiteKind {
    /// A position in the residual stream after a layer.
    Residual,
    /// An MLP hidden (post-non-linearity) neuron.
    Mlp,
}

impl FromStr for SiteKind {
    type Err = SteerError;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "residual" => Ok(SiteKind::Residual),
            "mlp" => Ok(SiteKind::Mlp),
            _ => Err(SteerError::InvalidSiteKind(s.to_string())),
        }
    }
}

/// A concrete (layer, index) coordinate inside the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Site {
    pub kind: SiteKind,
    pub layer: usize,
    pub index: usize,
}

impl Site {
    /// Validate the site against the model dimensions.
    pub fn validate(&self, layers: usize, hidden_dim: usize, mlp_dim: usize) -> Result<()> {
        if self.layer >= layers {
            return Err(SteerError::InvalidLayer {
                layer: self.layer,
                max: layers.saturating_sub(1),
            });
        }
        let dim = match self.kind {
            SiteKind::Residual => hidden_dim,
            SiteKind::Mlp => mlp_dim,
        };
        if self.index >= dim {
            return Err(SteerError::InvalidNeuron {
                layer: self.layer,
                neuron: self.index,
            });
        }
        Ok(())
    }
}

/// A single, replayable forward-pass edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Intervention {
    pub site: Site,
    pub mode: Mode,
    pub dose: f32,
}

impl Intervention {
    /// Convert to a runtime hook.
    #[must_use]
    pub fn to_hook(&self) -> Hook {
        Hook {
            site: self.site,
            mode: self.mode,
            dose: self.dose,
        }
    }

    /// Bound the dose magnitude to avoid trivial corruption.
    #[must_use]
    pub fn with_clipped_dose(mut self, max_abs: f32) -> Self {
        self.dose = self.dose.clamp(-max_abs, max_abs);
        self
    }
}
