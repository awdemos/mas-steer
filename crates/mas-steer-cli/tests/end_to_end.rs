//! End-to-end smoke test for the mas-steer CLI.

use std::process::Command;

#[test]
fn end_to_end_search_and_eval() {
    let tmp = tempfile::tempdir().unwrap();
    let champion = tmp.path().join("champion.json");
    let checkpoints = tmp.path().join("checkpoints");
    let prompts = tmp.path().join("prompts.txt");
    std::fs::write(
        &prompts,
        "Tell me a happy story.
What is the capital of Japan?
",
    )
    .unwrap();

    let search = Command::new(env!("CARGO_BIN_EXE_mas_steer"))
        .args([
            "search",
            "--model",
            "mock",
            "--generations",
            "3",
            "--population",
            "6",
            "--tasks",
            "4",
            "--controls",
            "2",
            "--checkpoint-dir",
            checkpoints.to_str().unwrap(),
            "--output",
            champion.to_str().unwrap(),
        ])
        .status()
        .expect("mas_steer search failed");
    assert!(search.success(), "search command failed");
    assert!(champion.exists(), "champion file missing");

    // Verify at least one checkpoint was written.
    let entries: Vec<_> = std::fs::read_dir(&checkpoints)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("checkpoint-"))
        .collect();
    assert!(!entries.is_empty(), "no checkpoint files written");

    let eval = Command::new(env!("CARGO_BIN_EXE_mas_steer"))
        .args([
            "eval",
            "--model",
            "mock",
            "--champion",
            champion.to_str().unwrap(),
            "--prompts",
            prompts.to_str().unwrap(),
        ])
        .status()
        .expect("mas_steer eval failed");
    assert!(eval.success(), "eval command failed");
}
