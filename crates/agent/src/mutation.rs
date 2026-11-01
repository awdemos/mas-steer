//! Variation operators for the proposer population.

use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;

use steer_core::{Intervention, Mode, Site, SiteKind};

/// Bounds for mutation.
#[derive(Debug, Clone, Copy)]
pub struct MutationBounds {
    pub layers: usize,
    pub hidden_dim: usize,
    pub mlp_dim: usize,
    pub max_interventions: usize,
    pub dose_range: (f32, f32),
}

impl MutationBounds {
    fn random_site(&self, rng: &mut impl Rng) -> Site {
        let kind = *[SiteKind::Residual, SiteKind::Mlp].choose(rng).unwrap();
        let dim = match kind {
            SiteKind::Residual => self.hidden_dim,
            SiteKind::Mlp => self.mlp_dim,
        };
        let layer = rng.gen_range(0..self.layers);
        let index = rng.gen_range(0..dim);
        Site { kind, layer, index }
    }
}

/// Randomly initialize a candidate.
pub fn random_candidate(bounds: &MutationBounds, rng: &mut impl Rng) -> Vec<Intervention> {
    let n = rng.gen_range(1..=bounds.max_interventions.clamp(1, 4));
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let site = bounds.random_site(rng);
        let mode = *[Mode::Add, Mode::Scale, Mode::Clamp, Mode::Zero]
            .choose(rng)
            .unwrap();
        let dose = rng.gen_range(bounds.dose_range.0..=bounds.dose_range.1);
        out.push(Intervention { site, mode, dose });
    }
    out
}

/// Mutate an existing candidate.
pub fn mutate(
    candidate: &[Intervention],
    bounds: &MutationBounds,
    rng: &mut impl Rng,
) -> Vec<Intervention> {
    let mut out = candidate.to_vec();
    if candidate.is_empty() || (candidate.len() < bounds.max_interventions && rng.gen_bool(0.5)) {
        out.push(Intervention {
            site: bounds.random_site(rng),
            mode: *[Mode::Add, Mode::Scale, Mode::Clamp, Mode::Zero]
                .choose(rng)
                .unwrap(),
            dose: rng.gen_range(bounds.dose_range.0..=bounds.dose_range.1),
        });
    } else {
        let idx = rng.gen_range(0..out.len());
        let roll = rng.gen_range(0..3);
        match roll {
            0 => {
                out[idx].site = bounds.random_site(rng);
            }
            1 => {
                out[idx].mode = *[Mode::Add, Mode::Scale, Mode::Clamp, Mode::Zero]
                    .choose(rng)
                    .unwrap();
            }
            _ => {
                let uni = Uniform::new_inclusive(bounds.dose_range.0, bounds.dose_range.1);
                out[idx].dose = uni.sample(rng);
            }
        }
    }
    out
}

/// One-point crossover between two parents.
pub fn crossover(
    a: &[Intervention],
    b: &[Intervention],
    bounds: &MutationBounds,
    rng: &mut impl Rng,
) -> Vec<Intervention> {
    let mut child = if a.is_empty() {
        b.to_vec()
    } else if b.is_empty() {
        a.to_vec()
    } else {
        let split_a = rng.gen_range(0..=a.len());
        let split_b = rng.gen_range(0..=b.len());
        let mut c = Vec::with_capacity(split_a + b.len() - split_b);
        c.extend_from_slice(&a[..split_a]);
        c.extend_from_slice(&b[split_b..]);
        c
    };
    if child.is_empty() {
        child.push(Intervention {
            site: bounds.random_site(rng),
            mode: *[Mode::Add, Mode::Scale, Mode::Clamp, Mode::Zero]
                .choose(rng)
                .unwrap(),
            dose: rng.gen_range(bounds.dose_range.0..=bounds.dose_range.1),
        });
    }
    child
}
