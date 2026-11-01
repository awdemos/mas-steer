//! Fitness computation for candidate interventions.

use serde::{Deserialize, Serialize};
use steer_core::Intervention;

/// Weighting constants for the fitness function.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FitnessConfig {
    /// Penalty on total intervention strength.
    pub lambda_str: f32,
    /// Penalty on control-task degradation.
    pub lambda_ctrl: f32,
    /// Maximum allowed per-token KL-divergence-like damage budget.
    /// A value of 0.0 disables the budget check.
    pub max_damage_budget: f32,
}

impl Default for FitnessConfig {
    fn default() -> Self {
        Self {
            lambda_str: 0.01,
            lambda_ctrl: 0.5,
            max_damage_budget: 0.0,
        }
    }
}

/// A fully populated score for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub task_success: f32,
    pub strength_penalty: f32,
    pub control_penalty: f32,
    pub fitness: f32,
}

impl Score {
    /// Compute fitness from raw components.
    pub fn from_components(
        task_success: f32,
        strength_penalty: f32,
        control_penalty: f32,
        config: &FitnessConfig,
    ) -> Self {
        let mut fitness = task_success
            - config.lambda_str * strength_penalty
            - config.lambda_ctrl * control_penalty;
        if config.max_damage_budget > 0.0 && control_penalty > config.max_damage_budget {
            fitness -= (control_penalty - config.max_damage_budget) * 10.0;
        }
        Self {
            task_success,
            strength_penalty,
            control_penalty,
            fitness: fitness.clamp(-10.0, 10.0),
        }
    }
}

/// Total absolute dose across a candidate set.
pub fn total_strength(interventions: &[Intervention]) -> f32 {
    interventions.iter().map(|i| i.dose.abs()).sum()
}

/// Control-task degradation as the drop in score between unsteered and steered.
pub fn control_penalty(control_baseline: f32, control_steered: f32) -> f32 {
    (control_baseline - control_steered).max(0.0)
}
