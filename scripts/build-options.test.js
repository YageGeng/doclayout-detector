import { readFile } from "node:fs/promises";
import test from "node:test";
import assert from "node:assert/strict";

test("wasm release build uses Rust opt-level 3", async () => {
  const buildScript = await readFile("scripts/build.js", "utf8");

  assert.match(buildScript, /CARGO_PROFILE_RELEASE_OPT_LEVEL/);
  assert.match(buildScript, /CARGO_PROFILE_RELEASE_OPT_LEVEL:\s*"3"/);
  assert.match(buildScript, /"backend-webgpu,wasm"/);
});
