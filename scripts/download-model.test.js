import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import assert from "node:assert/strict";

import {
  buildModelUrl,
  defaultOptions,
  ensureSafeTensorsFile,
  resolveOptions,
} from "./download-model.js";

test("buildModelUrl points at the Hugging Face resolve endpoint", () => {
  const url = buildModelUrl(
    "https://huggingface.co",
    "PaddlePaddle/PP-DocLayoutV3_safetensors",
    "model.safetensors",
    "main",
  );

  assert.equal(
    url,
    "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors/resolve/main/model.safetensors",
  );
});

test("resolveOptions keeps the project model path as the default output", () => {
  const options = resolveOptions([], {});

  assert.equal(
    options.output,
    join("models", "pp_doclayout_v3", "model.safetensors"),
  );
  assert.equal(options.repo, defaultOptions.repo);
  assert.equal(options.filename, defaultOptions.filename);
  assert.equal(options.force, false);
});

test("resolveOptions supports the output path and force flags", () => {
  const options = resolveOptions(["--output", "tmp/model.safetensors", "--force"], {});

  assert.equal(options.output, "tmp/model.safetensors");
  assert.equal(options.force, true);
});

test("ensureSafeTensorsFile accepts a minimal safetensors header", async () => {
  const dir = await mkdtemp(join(tmpdir(), "doclayout-model-test-"));
  const file = join(dir, "model.safetensors");

  try {
    // The safetensors format starts with an eight-byte little-endian header length.
    const header = Buffer.from('{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}');
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(header.length));
    await writeFile(file, Buffer.concat([length, header, Buffer.alloc(4)]));

    await ensureSafeTensorsFile(file);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("ensureSafeTensorsFile rejects non-safetensors content", async () => {
  const dir = await mkdtemp(join(tmpdir(), "doclayout-model-test-"));
  const file = join(dir, "model.safetensors");

  try {
    await writeFile(file, "not a safetensors file");

    await assert.rejects(
      () => ensureSafeTensorsFile(file),
      /invalid safetensors header/i,
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
