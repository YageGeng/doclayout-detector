use std::collections::BTreeMap;

#[test]
fn package_name_matches_doclayout_detector_project() {
    assert_eq!(
        parse_package_name(include_str!("../Cargo.toml")),
        "doclayout-detector"
    );
}

#[test]
fn features_match_backend_cli_and_webgpu_shape() {
    let features = parse_cargo_features(include_str!("../Cargo.toml"));

    assert_feature_items(&features, "default", &["backend-webgpu", "cli"]);
    assert_feature_items(&features, "cli", &["dep:clap", "dep:png"]);
    assert_feature_items(
        &features,
        "backend-webgpu",
        &["burn-wgpu/webgpu", "burn-wgpu/metal", "dep:burn-wgpu"],
    );
    assert!(!features.contains_key("native-cli"));
    assert!(!features.contains_key("backend-ndarray"));
    assert_feature_contains(&features, "wasm", "backend-webgpu");
    assert_feature_contains(&features, "wasm", "dep:wasm-bindgen");
    assert_feature_contains(&features, "wasm", "dep:tracing-wasm");
    assert!(!features["backend-webgpu"].contains(&"panic_hook".to_string()));
    assert!(!features["backend-webgpu"].contains(&"dep:wasm-bindgen".to_string()));
}

#[test]
fn native_webgpu_backend_uses_auto_graphics_api() {
    let model = include_str!("../src/model.rs");

    assert!(model.contains("const BACKEND_NAME: &str = \"auto\""));
    assert!(model.contains("const BACKEND_NAME: &str = \"webgpu\""));
    assert!(model.contains("init_setup::<burn_wgpu::graphics::AutoGraphicsApi>"));
    assert!(model.contains("init_setup_async::<burn_wgpu::graphics::WebGpu>"));
}

#[test]
fn native_webgpu_backend_uses_metal_safe_topk_path() {
    let model = include_str!("../src/pp_doclayout/model.rs");

    assert!(model.contains("feature = \"backend-metal\""));
    assert!(model.contains("feature = \"backend-webgpu\""));
    assert!(model.contains("target_os = \"macos\""));
    assert!(model.contains("scores.topk_with_indices(topk, 1).1"));
}

fn assert_feature_items(
    features: &BTreeMap<String, Vec<String>>,
    feature: &str,
    expected: &[&str],
) {
    let actual = features
        .get(feature)
        .unwrap_or_else(|| panic!("missing feature {feature}"));
    assert_eq!(actual, expected);
}

fn assert_feature_contains(features: &BTreeMap<String, Vec<String>>, feature: &str, item: &str) {
    let actual = features
        .get(feature)
        .unwrap_or_else(|| panic!("missing feature {feature}"));
    assert!(
        actual.iter().any(|actual| actual == item),
        "feature {feature} should include {item}, got {actual:?}"
    );
}

fn parse_package_name(manifest: &str) -> &str {
    let mut in_package = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if in_package && trimmed.starts_with('[') {
            break;
        }
        if in_package && trimmed.starts_with("name") {
            let (_, value) = trimmed.split_once('=').expect("package name should use =");
            return value.trim().trim_matches('"');
        }
    }
    panic!("missing package name")
}

fn parse_cargo_features(manifest: &str) -> BTreeMap<String, Vec<String>> {
    let mut features = BTreeMap::new();
    let mut in_features = false;
    let mut lines = manifest.lines().peekable();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed == "[features]" {
            in_features = true;
            continue;
        }
        if in_features && trimmed.starts_with('[') {
            break;
        }
        if !in_features || trimmed.is_empty() {
            continue;
        }

        let Some((name, value)) = trimmed.split_once('=') else {
            continue;
        };
        let name = name.trim().to_string();
        let mut value = value.trim().to_string();
        while value.contains('[') && !value.contains(']') {
            let Some(next_line) = lines.next() else {
                break;
            };
            value.push_str(next_line.trim());
        }
        features.insert(name, parse_string_array(&value));
    }

    features
}

fn parse_string_array(value: &str) -> Vec<String> {
    value
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(',')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|item| {
            let item = item.trim().trim_matches('"');
            (!item.is_empty()).then(|| item.to_string())
        })
        .collect()
}
