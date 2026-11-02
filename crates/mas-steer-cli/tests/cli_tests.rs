use std::process::Command;

#[test]
fn cli_search_smoke() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("champion.json");
    let status = Command::new(env!("CARGO_BIN_EXE_mas_steer"))
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
            "--output",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run mas-steer");
    assert!(status.success());
    assert!(out.exists());
}
