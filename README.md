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
| `steer-core` | `Backend` trait, interventions, hooks, real Candle model loader, mock backend |
| `arena` | Task families, challenger prompt generation, fitness scoring |
| `agent` | Genetic search loop, population variation, parallel evaluation |
| `mas-steer-cli` | Command-line interface (`mas_steer`) |

## Build

```bash
cargo build --workspace --release
```

The default build compiles the real Candle-backed model loader (Qwen2 family). A
fast mock backend is available for tests and offline development by building with
the `mock` feature:

```bash
cargo build --workspace --features mock
```

## Stage or download a model

Use the CLI to pull a Qwen2-compatible checkpoint from Hugging Face Hub:

```bash
cargo run --bin mas_steer -- pull Qwen/Qwen2.5-0.5B --output-dir ./models
```

Or point the tool at any local directory containing `config.json`,
`model.safetensors`, and `tokenizer.json`.

## Run a search

```bash
cargo run --bin mas_steer -- search \
  --model ./models/Qwen--Qwen2.5-0.5B \
  --generations 10 --population 20 \
  --checkpoint-dir checkpoints \
  --output champion.json
```

Use a config file to set search hyperparameters and the damage budget:

```bash
cargo run --bin mas_steer -- search \
  --model ./models/Qwen--Qwen2.5-0.5B \
  --config examples/config.toml
```

## Resume a search

```bash
cargo run --bin mas_steer -- resume \
  --model ./models/Qwen--Qwen2.5-0.5B \
  --checkpoint checkpoints/checkpoint-0008.json \
  --generations 10 --output champion.json
```

## Evaluate a saved champion

```bash
cargo run --bin mas_steer -- eval \
  --model ./models/Qwen--Qwen2.5-0.5B \
  --champion champion.json --prompts prompts.txt
```

## Mock-only smoke test

For CI or machines without a model checkpoint, use the mock backend:

```bash
cargo run --bin mas_steer --features mock -- search \
  --model mock --generations 3 --population 6
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
