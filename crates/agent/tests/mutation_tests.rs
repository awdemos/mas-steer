use agent::mutation::{crossover, mutate, random_candidate, MutationBounds};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bounds() -> MutationBounds {
    MutationBounds {
        layers: 4,
        hidden_dim: 16,
        mlp_dim: 32,
        max_interventions: 4,
        dose_range: (-5.0, 5.0),
    }
}

#[test]
fn random_candidate_is_non_empty_and_valid() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let c = random_candidate(&bounds(), &mut rng);
    assert!(!c.is_empty());
    assert!(c.len() <= 4);
    for iv in &c {
        assert!(iv.dose.abs() <= 5.0);
        assert!(iv.site.layer < 4);
    }
}

#[test]
fn mutate_preserves_non_emptiness() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let c = random_candidate(&bounds(), &mut rng);
    let m = mutate(&c, &bounds(), &mut rng);
    assert!(!m.is_empty());
}

#[test]
fn crossover_never_returns_empty() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let a = random_candidate(&bounds(), &mut rng);
    let b = random_candidate(&bounds(), &mut rng);
    let child = crossover(&a, &b, &bounds(), &mut rng);
    assert!(!child.is_empty());
}
