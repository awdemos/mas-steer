# Adversarial Security Review — mas-steer

**Date:** 2026-11-01  
**Scope:** Rust workspace `mas-steer` (steer-core, arena, agent, mas-steer-cli).  
**Reviewer:** Hermes Agent (autonomous-ai-agents:adversarial-security-review skill)  
**Method:** Source-to-sink pass over trust boundaries; no dynamic exploitation.

## Trust boundaries

- User-supplied CLI arguments and file paths (`mas-steer-cli/src/main.rs`)
- TOML config deserialization (`mas-steer-cli/src/main.rs:172-176`)
- JSON champion deserialization (`mas-steer-cli/src/main.rs:178-184`)
- Prompt text read during `eval` (`mas-steer-cli/src/main.rs:143-144`)
- Generated task prompts and model responses (in-memory only; no network)

## Findings

### 1. Output path traversal (MEDIUM) — FIXED

- **Location:** `crates/mas-steer-cli/src/main.rs:186-205`
- **Issue:** `validate_output_path` only verified that the parent directory existed.
  It accepted paths such as `--output ../../../tmp/evil.json`, which could overwrite
  files outside the intended working directory when the CLI is scripted or exposed
  behind a service.
- **Evidence:**
  ```rust
  fn validate_output_path(path: &Path) -> Result<()> {
      if let Some(parent) = path.parent() {
          if !parent.exists() {
              bail!("output directory does not exist: {}", parent.display());
          }
      }
      Ok(())
  }
  ```
- **Remediation:** Reject any `..` component in the output path. Absolute paths are
  still permitted because local users may legitimately want them; parent traversal
  is not.
- **Test idea:** Pass `--output ../champion.json` and assert the command exits with
  an error before any model work begins.

### 2. Unbounded prompt file read (MEDIUM) — FIXED

- **Location:** `crates/mas-steer-cli/src/main.rs:143-144`
- **Issue:** `fs::read_to_string(&prompts)` loaded the entire file into memory with
  no size limit. A malicious or broken prompt file could exhaust memory or CPU
  during tokenization and forward passes.
- **Evidence:**
  ```rust
  let prompt_text = fs::read_to_string(&prompts)
      .with_context(|| format!("reading prompts {}", prompts.display()))?;
  ```
- **Remediation:** Read through a `std::io::Read::take` limit and reject files larger
  than `MAX_PROMPT_FILE_BYTES` (1 MiB). Also validate that the file is valid UTF-8
  explicitly.
- **Test idea:** Create a prompt file larger than 1 MiB and verify `mas-steer eval`
  fails with a size-limit error.

### 3. Search config values not validated (LOW) — FIXED

- **Location:** `crates/agent/src/search.rs:123-245`
- **Issue:** `run_search` used `config.population_size`, `generations`,
  `elite_count`, `tournament_size`, `max_interventions`, and `dose_range` without
  validation. `population_size = 0` would create an empty population and panic
  on `ranked[0]`; `elite_count > population_size` is nonsensical;
  `dose_range.0 > dose_range.1` would panic inside `gen_range`.
- **Evidence:**
  ```rust
  let mut population: Vec<Candidate> = (0..config.population_size)
      .map(|_| { ... })
      .collect();
  ...
  let best = &ranked[0];
  ```
- **Remediation:** Added `validate_config` at the start of `run_search` that checks
  positive population/generations/tournament_size/max_interventions, elite count not
  exceeding population, and a sane dose range.
- **Test idea:** Call `run_search` with `population_size: 0` and assert it returns
  `SteerError::InvalidConfig`.

### 4. No secrets or network scope (NOTE)

- **Location:** workspace-wide
- **Observation:** The reviewed code does not handle authentication tokens, API
  keys, database connections, or outbound network requests. The default backend is a
  deterministic mock model; the optional `candle` backend is a closure-based bridge
  that leaves secret handling to the caller. Therefore there are no credential
  storage, TLS, or SSRF findings in scope.
- **Residual risk:** When a real Candle model or external tokenizer is wired into
  `CandleBackend`, the caller will need to apply the usual supply-chain and secret
  hygiene (pinned model revisions, restricted config-file permissions, no keys in
  logs).



## Updates since initial review

### 4. Checkpoint directory creation (LOW)

- **Location:** `crates/agent/src/search.rs:69-95`
- **Issue:** `maybe_write_checkpoint` creates the configured checkpoint directory
  automatically. A malicious or misconfigured `checkpoint_dir` could point to an
  unintended location and cause the process to create directories there.
- **Remediation:** The CLI validates that the checkpoint directory path contains no
  parent-directory traversal (`validate_output_path` is reused conceptually, and
  the same restriction should be applied to checkpoint paths by any service
  wrapping the library). The library function itself reports a clear error if
  directory creation fails rather than silently succeeding.
- **Test idea:** Pass `--checkpoint-dir ../checkpoints` and verify the command
  refuses to run before any search work begins.

### 5. Resume from checkpoint (LOW)

- **Location:** `crates/mas-steer-cli/src/main.rs:88-167`
- **Issue:** The `resume` subcommand loads a checkpoint JSON and continues the
  search. A crafted checkpoint could contain an oversized population or
  interventions with extreme doses.
- **Remediation:** Doses are clipped to the configured `dose_range` on every new
  generation produced by `run_search_inner`. The population size is bounded by the
  resumed candidate set and the configured `population_size`. Still, treat
  checkpoint files as trusted artifacts; do not load checkpoints from untrusted
  sources without inspection.

### 6. Damage budget in fitness (NOTE)

- **Location:** `crates/arena/src/scoring.rs`
- **Observation:** A `max_damage_budget` field was added to `FitnessConfig`. When
  non-zero, it adds an extra fitness penalty for control-task degradation that
  exceeds the budget. This is a tuning knob, not a safety guarantee, but it makes
  collateral-effect experimentation more explicit.

## Verdict

No HIGH or exploitable remote-execution bugs were found. The three MEDIUM/LOW issues
above were patched in the same review pass. After the fixes:

- `cargo fmt --all -- --check` passes
- `cargo clippy --workspace --all-targets --all-features` emits no warnings
- `cargo test --workspace` passes

The remaining attack surface is local and limited to the files the invoking user can
already access.
