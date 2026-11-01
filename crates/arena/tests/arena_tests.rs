use arena::{Challenger, FitnessConfig, Score, TaskFamily};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn challenger_generates_tasks() {
    let c = Challenger::new();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let tasks = c.generate_tasks(20, &mut rng);
    assert!(!tasks.is_empty());
    assert!(tasks
        .iter()
        .any(|t| t.family == TaskFamily::SentimentPositive));
}

#[test]
fn score_fitness_degrades_with_strength() {
    let config = FitnessConfig::default();
    let s = Score::from_components(1.0, 10.0, 0.0, &config);
    assert!(s.fitness < 1.0);
}
