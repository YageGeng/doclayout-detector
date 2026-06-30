#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { rmSync } from "node:fs";

rmSync("pkg", { recursive: true, force: true });

const wasmPackArgs = [
  "build",
  ".",
  "--release",
  "--target",
  "web",
  "--out-dir",
  "pkg",
  "--out-name",
  "doclayout_detector",
  "--",
  "--no-default-features",
  "--features",
  "backend-webgpu,wasm",
];

const wasmTargetRustflags = [
  process.env.CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS,
  "-C link-arg=--allow-undefined",
]
  .filter(Boolean)
  .join(" ");

const build = spawnSync("wasm-pack", wasmPackArgs, {
  env: {
    ...process.env,
    // Force Rust's release profile to use O3 for the wasm target build.
    CARGO_PROFILE_RELEASE_OPT_LEVEL: "3",
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS: wasmTargetRustflags,
  },
  stdio: "inherit",
});

if (build.status !== 0) {
  process.exit(build.status ?? 1);
}

const patch = spawnSync("node", ["scripts/patch-wasi-imports.js"], {
  stdio: "inherit",
});

process.exit(patch.status ?? 1);
