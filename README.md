# MAS-Steer

Multi-Agent Search over neuron intervention spaces, with self-adversarial arenas.

This repository implements the system described in the paper at
[neuronengineering.dev/papers/mas-steer/](https://neuronengineering.dev/papers/mas-steer/).
It is a Rust workspace that searches for steering interventions in frozen
language models, evaluates them against evolving adversarial tasks, and archives
the winners as replayable JSON artifacts.

## Crates

| Crate | Responsibility |
|-------|----------------|
| `steer-core` | `Backend` trait, interventions, hooks, mock backend, optional Candle bridge |
| `arena` | Task families, challenger prompt generation, fitness scoring |
| `agent` | Genetic search loop, population variation, parallel evaluation |
| `mas-steer-cli` | Command-line interface (`mas-steer`) |

## Build

```bash
cargo build --workspace --release
```

The default build uses a fast mock backend. The optional Candle integration
is disabled by default; enable it only when you are ready to wire in a real model. To enable the optional Candle
integration, build with `--features candle` on `steer-core` and the CLI.

## Run a search

```bash
cargo run --bin mas_steer -- search \
  --layers 6 --hidden-dim 32 --mlp-dim 64 --vocab-size 128 \
  --generations 10 --population 20 \
  --checkpoint-dir checkpoints \
  --output champion.json
```

Use a config file to set search hyperparameters and the optional damage budget:

```bash
cargo run --bin mas_steer -- search --config examples/config.toml
```

## Resume a search

```bash
cargo run --bin mas_steer -- resume \
  --checkpoint checkpoints/checkpoint-0008.json \
  --generations 10 --output champion.json
```

## Evaluate a saved champion

```bash
cargo run --bin mas_steer -- eval \
  --champion champion.json --prompts prompts.txt
```

## Verification gates

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace
```

## Security

See `reports/adversarial-mas-steer-2026-11-01.md` for the latest adversarial
security review. Security issues should be reported to
hello@neuronengineering.dev.

## License

MIT — see `LICENSE`.
