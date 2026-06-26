import { readFile } from "node:fs/promises";
import test from "node:test";
import assert from "node:assert/strict";

test("wasm model inference exposes detailed profiling events", async () => {
  const model = await readFile("src/pp_doclayout/model.rs", "utf8");
  const embedded = await readFile("src/model.rs", "utf8");

  for (const step of [
    "forward_async_encode",
    "encode_backbone",
    "encode_encoder",
    "forward_async_topk",
    "proposal_topk_iterative",
    "prepare_decoder_from_topk",
    "forward_decoder",
    "forward_order_head",
  ]) {
    assert.match(model, new RegExp(step));
  }

  for (const step of [
    "input_upload",
    "logits_readback",
    "pred_boxes_readback",
    "order_logits_readback",
  ]) {
    assert.match(embedded, new RegExp(step));
  }
});

test("wasm proposal top-k stays on WebGPU without slow argtopk or sorting", async () => {
  const model = await readFile("src/pp_doclayout/model.rs", "utf8");
  // Burn WebGPU sorting panics and argtopk is too slow, so this path must stay on GPU with iterative argmax.
  const asyncTopk = model.match(
    /async fn proposal_topk_indices_async[\s\S]*?\n}\n\n#\[cfg\(test\)\]/,
  )?.[0];

  assert.ok(asyncTopk, "proposal_topk_indices_async should exist");
  assert.match(asyncTopk, /gpu_topk_indices/);
  assert.doesNotMatch(asyncTopk, /into_data_async/);
  assert.doesNotMatch(asyncTopk, /host_topk_indices_from_values/);
  assert.doesNotMatch(asyncTopk, /argtopk/);
  assert.doesNotMatch(asyncTopk, /topk_with_indices/);
});

test("wasm exposes loaded page batch detection", async () => {
  const wasm = await readFile("src/wasm.rs", "utf8");
  const detector = await readFile("src/pp_doclayout/detector.rs", "utf8");

  assert.match(wasm, /js_name = detectLoadedPages/);
  assert.match(wasm, /detect_loaded_pages/);
  assert.match(wasm, /detect_rendered_pages/);
  assert.match(detector, /detect_pages_async/);
  assert.match(detector, /infer_batch_async/);
});
