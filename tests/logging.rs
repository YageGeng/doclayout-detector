use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn rust_code_uses_tracing_instead_of_printing() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();

    collect_forbidden_logging(&root.join("src"), &mut violations);
    collect_forbidden_logging(&root.join("tests"), &mut violations);

    assert!(
        violations.is_empty(),
        "use tracing instead of print/eprint/dbg macros:\n{}",
        violations.join("\n")
    );
}

fn collect_forbidden_logging(path: &Path, violations: &mut Vec<String>) {
    if path.is_dir() {
        for entry in fs::read_dir(path).expect("failed to read directory") {
            let entry = entry.expect("failed to read directory entry");
            collect_forbidden_logging(&entry.path(), violations);
        }
        return;
    }

    if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
        return;
    }

    let contents = fs::read_to_string(path).expect("failed to read Rust source");
    let relative = path
        .strip_prefix(env!("CARGO_MANIFEST_DIR"))
        .unwrap_or(path);
    for (line_index, line) in contents.lines().enumerate() {
        if contains_forbidden_logging_macro(line) {
            violations.push(format!("{}:{}", relative.display(), line_index + 1));
        }
    }
}

fn contains_forbidden_logging_macro(line: &str) -> bool {
    [
        concat!("print", "ln!"),
        concat!("eprint", "ln!"),
        concat!("print", "!"),
        concat!("eprint", "!"),
        concat!("dbg", "!"),
    ]
    .iter()
    .any(|needle| line.contains(needle))
}
