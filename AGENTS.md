# AGENTS.md — Developer / Agent Guide for MAS-Steer

This guide is for AI agents, CI systems, and human contributors working on the
`mas-steer` workspace. Read `README.md` first for user-facing behavior, then
use this file for development conventions and safe operation.

## Repository layout

```
mas-steer/
├── Cargo.toml              # workspace root
├── deny.toml               # cargo-deny policy
├── .cargo/audit.toml         # cargo-audit policy
├── crates/
│   ├── steer-core/         # Backend trait, interventions, mock backend
│   ├── arena/              # Task families, challenger prompts, scoring
│   ├── agent/              # Genetic search loop
│   └── mas-steer-cli/      # CLI binary: mas_steer
├── reports/                # adversarial security review artifacts
├── tests/                  # workspace-level integration tests (if added later)
└── AGENTS.md               # this file
```

## Environment assumptions

- Linux or macOS development host.
- Rust stable toolchain (see `rust-version` in workspace `Cargo.toml`).
- `cargo`, `cargo-fmt`, `cargo-clippy` available.
- Optional but recommended: `cargo-audit`, `cargo-deny`.
- No GPU or special drivers needed for the default mock backend.

## Build

```bash
cargo build --workspace --release
```

To build with the optional Candle backend:

```bash
cargo build --workspace --features candle
```

Note that enabling `candle` pulls in a large dependency graph and is not needed
for the algorithmic test suite.

## Run tests

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
```

Run integration-style smoke tests:

```bash
cargo run --bin mas_steer -- search \
  --layers 4 --hidden-dim 16 --mlp-dim 32 --vocab-size 64 \
  --generations 3 --population 6 --tasks 4 --controls 2 \
  --output /tmp/smoke-champion.json

cargo run --bin mas_steer -- eval \
  --champion /tmp/smoke-champion.json \
  --prompts /dev/stdin <<'EOF'
Tell me a happy story.
EOF
```

## Adding a new task family

1. Add the family variant to `crates/arena/src/task.rs` (`TaskFamily`).
2. Implement a `Display` branch and a `score` branch.
3. Add templates in `crates/arena/src/challenger.rs`.
4. Add a unit test in `crates/arena/tests/arena_tests.rs`.
5. Run `cargo test -p arena`.

## Adding a new intervention mode

1. Add the variant to `crates/steer-core/src/intervention.rs` (`Mode`).
2. Implement `Mode::apply`.
3. Update `Mode::from_str` and any exhaustive `match` sites.
4. Add a unit test in `crates/steer-core/tests/backend_tests.rs`.

## Adding a real model backend

The current `CandleBackend` is a thin closure bridge. To plug in a concrete
Candle model (for example a Qwen or Llama implementation from
`candle-transformers`):

1. Create a new crate `crates/steer-candle-models`.
2. Depend on `steer-core` (path) and `candle-transformers`.
3. Implement `Backend` for a model-specific wrapper.
4. Add an optional feature on `mas-steer-cli` that wires the new backend into
the `search` and `eval` commands.

Do **not** place model weights, API keys, or HF tokens in this repository.

## Conventions

- Use `anyhow::Result` at the CLI boundary and `steer_core::Result` inside
  `steer-core`. Other crates use `steer_core::Result` for steering errors.
- Keep dependencies per-crate minimal. The default build must not require CUDA.
- Prefer deterministic tests with fixed seeds; mock the backend for unit tests.
- Never use `unsafe` code (`unsafe_code = "forbid"` is set at workspace level).
- File writes in the CLI must validate paths against parent-directory traversal.
- Prompt and config reads must be size-bounded.

## Common pitfalls

- `run_search` panics on `ranked[0]` if `population_size` is 0. This is guarded
  by `validate_config`, but never call the inner loop with an empty population.
- The default backend is a toy model; it proves the search works, not that a
  steering vector transfers to a real model.
- Generated prompts may contain refusal-like content because challengers
  intentionally vary task families. This is by design for the self-adversarial
  arena; see `SAFETY.md` for how to run responsibly.

## CI checklist

For every PR, run:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
cargo audit
cargo deny check
```

If any of the audit/deny checks fail, fix them before merging. If a dependency
advisory cannot be immediately resolved, document the exception in `reports/` and
file a follow-up issue.

## Where to ask

Open an issue at `https://github.com/awdemos/mas-steer` or email
hello@neuronengineering.dev.
