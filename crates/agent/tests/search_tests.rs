use std::sync::Arc;

use agent::{run_search, SearchConfig};
use arena::{TaskFamily, TaskInstance};
use steer_core::{MockBackend, Mode, ModelDims, Site, SiteKind};

#[test]
fn search_finds_champion() {
    let backend: Arc<dyn steer_core::Backend> = Arc::new(MockBackend::new(
        ModelDims {
            layers: 6,
            hidden_dim: 32,
            mlp_dim: 64,
            vocab_size: 128,
        },
        42,
    ));
    let config = SearchConfig {
        generations: 5,
        population_size: 10,
        ..SearchConfig::default()
    };
    let result = run_search(backend, config, 123, 8, 4).unwrap();
    assert!(!result.champion.candidate.interventions.is_empty());
}

#[test]
fn evaluate_candidate_changes_with_targeted_hook() {
    let backend: Arc<dyn steer_core::Backend> = Arc::new(MockBackend::new(
        ModelDims {
            layers: 4,
            hidden_dim: 16,
            mlp_dim: 32,
            vocab_size: 64,
        },
        42,
    ));
    let task = TaskInstance {
        family: TaskFamily::SentimentPositive,
        prompt: "Tell me a happy story.".to_string(),
    };
    let control = TaskInstance {
        family: TaskFamily::FactRecall,
        prompt: "What is the capital of Japan?".to_string(),
    };
    let empty = agent::Candidate {
        interventions: vec![],
    };
    let scored_empty = agent::evaluate_candidate(
        backend.as_ref(),
        &empty,
        std::slice::from_ref(&task),
        std::slice::from_ref(&control),
        &arena::FitnessConfig::default(),
    );

    let steered = agent::Candidate {
        interventions: vec![steer_core::Intervention {
            site: Site {
                kind: SiteKind::Residual,
                layer: 0,
                index: 0,
            },
            mode: Mode::Add,
            dose: 2.0,
        }],
    };
    let scored_steered = agent::evaluate_candidate(
        backend.as_ref(),
        &steered,
        std::slice::from_ref(&task),
        std::slice::from_ref(&control),
        &arena::FitnessConfig::default(),
    );

    // The empty and non-empty candidates should produce different scores.
    assert_ne!(scored_empty.fitness, scored_steered.fitness);
}
