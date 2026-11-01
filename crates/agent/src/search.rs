//! Multi-agent search loop over intervention spaces.

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

use arena::{control_penalty, total_strength, Challenger, FitnessConfig, Score, TaskInstance};
use steer_core::{Backend, Intervention, SteerError};

use crate::mutation::{crossover, mutate, random_candidate, MutationBounds};
use crate::population::{Candidate, Champion};

/// Configuration for the genetic search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub generations: usize,
    pub population_size: usize,
    pub elite_count: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub max_interventions: usize,
    pub dose_range: (f32, f32),
    pub fitness: FitnessConfig,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            generations: 10,
            population_size: 20,
            elite_count: 2,
            tournament_size: 3,
            mutation_rate: 0.6,
            max_interventions: 4,
            dose_range: (-5.0, 5.0),
            fitness: FitnessConfig::default(),
        }
    }
}

/// Result of a completed search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub champion: Champion,
    pub history: Vec<Score>,
    pub config: SearchConfig,
}

/// Evaluate one candidate on the current task and control sets.
pub fn evaluate_candidate(
    backend: &dyn Backend,
    candidate: &Candidate,
    tasks: &[TaskInstance],
    controls: &[TaskInstance],
    config: &FitnessConfig,
) -> Score {
    let mut task_scores = Vec::with_capacity(tasks.len());
    for task in tasks {
        let response = generate_response(backend, &task.prompt, &candidate.interventions);
        task_scores.push(task.score(&response));
    }
    let task_success = mean(&task_scores);

    let mut control_baseline = Vec::with_capacity(controls.len());
    let mut control_steered = Vec::with_capacity(controls.len());
    for ctrl in controls {
        let base = generate_response(backend, &ctrl.prompt, &[]);
        let steer = generate_response(backend, &ctrl.prompt, &candidate.interventions);
        control_baseline.push(ctrl.score(&base));
        control_steered.push(ctrl.score(&steer));
    }
    let baseline = mean(&control_baseline);
    let steered = mean(&control_steered);
    let ctrl_penalty = control_penalty(baseline, steered);
    let str_penalty = total_strength(&candidate.interventions);

    Score::from_components(task_success, str_penalty, ctrl_penalty, config)
}

/// Greedy argmax decoding of a short response.
pub fn generate_response(
    backend: &dyn Backend,
    prompt: &str,
    interventions: &[Intervention],
) -> String {
    let hooks: Vec<_> = interventions.iter().map(Intervention::to_hook).collect();
    let mut tokens = match backend.tokenize(prompt) {
        Ok(t) => t,
        Err(_) => return String::new(),
    };
    // Generate up to 8 tokens.
    for _ in 0..8 {
        let logits = match backend.forward(&tokens, &hooks) {
            Ok(l) => l,
            Err(_) => break,
        };
        let next = argmax(&logits);
        tokens.push(next);
    }
    backend.decode(&tokens).unwrap_or_default()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

/// Validate search configuration values before starting.
fn validate_config(config: &SearchConfig) -> steer_core::Result<()> {
    if config.population_size == 0 {
        return Err(SteerError::InvalidConfig(
            "population_size must be > 0".to_string(),
        ));
    }
    if config.generations == 0 {
        return Err(SteerError::InvalidConfig(
            "generations must be > 0".to_string(),
        ));
    }
    if config.elite_count > config.population_size {
        return Err(SteerError::InvalidConfig(
            "elite_count must be <= population_size".to_string(),
        ));
    }
    if config.tournament_size == 0 {
        return Err(SteerError::InvalidConfig(
            "tournament_size must be > 0".to_string(),
        ));
    }
    if config.max_interventions == 0 {
        return Err(SteerError::InvalidConfig(
            "max_interventions must be > 0".to_string(),
        ));
    }
    let (lo, hi) = config.dose_range;
    if lo > hi {
        return Err(SteerError::InvalidConfig(
            "dose_range lower bound must not exceed upper bound".to_string(),
        ));
    }
    Ok(())
}

/// Run the full multi-agent search.
pub fn run_search(
    backend: Arc<dyn Backend>,
    config: SearchConfig,
    seed: u64,
    task_count: usize,
    control_count: usize,
) -> Result<SearchResult, SteerError> {
    let dims = backend.dims();
    if dims.layers == 0 || dims.hidden_dim == 0 || dims.vocab_size == 0 {
        return Err(SteerError::Backend("invalid model dimensions".to_string()));
    }

    validate_config(&config)?;

    let bounds = MutationBounds {
        layers: dims.layers,
        hidden_dim: dims.hidden_dim,
        mlp_dim: dims.mlp_dim,
        max_interventions: config.max_interventions,
        dose_range: config.dose_range,
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let challenger = Challenger::new();

    // Generate task and control sets.
    let tasks = challenger.generate_tasks(task_count, &mut rng);
    let controls = challenger.generate_tasks(control_count, &mut rng);

    // Initialize population.
    let mut population: Vec<Candidate> = (0..config.population_size)
        .map(|_| {
            let ivs = random_candidate(&bounds, &mut rng);
            Candidate { interventions: ivs }.deduped()
        })
        .collect();

    let mut champion: Option<Champion> = None;
    let mut history = Vec::with_capacity(config.generations);

    for generation in 0..config.generations {
        info!("search generation {}", generation);

        // Evaluate population in parallel.
        let scored: Vec<(Candidate, Score)> = population
            .par_iter()
            .map(|c| {
                let score =
                    evaluate_candidate(backend.as_ref(), c, &tasks, &controls, &config.fitness);
                (c.clone(), score)
            })
            .collect();

        // Sort by fitness descending.
        let mut ranked = scored;
        ranked.sort_by(|a, b| {
            b.1.fitness
                .partial_cmp(&a.1.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = &ranked[0];
        history.push(best.1);
        debug!("gen {} best fitness {}", generation, best.1.fitness);

        if champion
            .as_ref()
            .map_or(true, |c| best.1.fitness > c.fitness)
        {
            champion = Some(Champion {
                candidate: best.0.clone(),
                generation,
                fitness: best.1.fitness,
            });
        }

        // Build next generation.
        let elite: Vec<Candidate> = ranked
            .iter()
            .take(config.elite_count.min(ranked.len()))
            .map(|(c, _)| c.clone())
            .collect();

        let mut next = elite.clone();
        while next.len() < config.population_size {
            let parent =
                tournament_select(&ranked, config.tournament_size.min(ranked.len()), &mut rng);
            let mut child_ivs = if rng.gen_bool(config.mutation_rate) {
                mutate(&parent.interventions, &bounds, &mut rng)
            } else if rng.gen_bool(0.5) {
                let other =
                    tournament_select(&ranked, config.tournament_size.min(ranked.len()), &mut rng);
                crossover(
                    &parent.interventions,
                    &other.interventions,
                    &bounds,
                    &mut rng,
                )
            } else {
                parent.interventions.clone()
            };
            // Clip doses to avoid runaway values.
            child_ivs = child_ivs
                .into_iter()
                .map(|i| i.with_clipped_dose(config.dose_range.1.abs()))
                .collect();
            next.push(
                Candidate {
                    interventions: child_ivs,
                }
                .deduped(),
            );
        }

        population = next;
    }

    let champion = champion.ok_or(SteerError::EmptyInterventionSet)?;
    Ok(SearchResult {
        champion,
        history,
        config,
    })
}

fn tournament_select<'a>(
    ranked: &'a [(Candidate, Score)],
    k: usize,
    rng: &mut impl Rng,
) -> &'a Candidate {
    let idxs: Vec<usize> = (0..ranked.len()).collect();
    let best = *idxs
        .choose_multiple(rng, k)
        .min_by(|a, b| {
            ranked[**b]
                .1
                .fitness
                .partial_cmp(&ranked[**a].1.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(&0);
    &ranked[best].0
}
