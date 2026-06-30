import { readFile } from "node:fs/promises";
import test from "node:test";
import assert from "node:assert/strict";

test("preview page defaults DPI to 96", async () => {
  const html = await readFile("index.html", "utf8");

  assert.match(html, /id="dpi"[^>]*value="96"/);
  assert.match(html, /Number\(dpiInput\.value \|\| 96\)/);
  assert.doesNotMatch(html, /value="144"/);
  assert.doesNotMatch(html, /dpiInput\.value \|\| 144/);
});

test("preview page detects the current page by default", async () => {
  const html = await readFile("index.html", "utf8");

  assert.match(html, /id="page"/);
  assert.match(html, /id="prev"/);
  assert.match(html, /id="next"/);
  assert.match(html, /id="run-all"/);
  assert.match(html, /detectCurrentPage/);
  assert.match(html, /detectAllPages/);
  assert.doesNotMatch(
    html,
    /runButton\.addEventListener\("click"[\s\S]*?for \(let pageNumber = 1; pageNumber <= count;/,
  );
});

test("preview page exposes batch progress controls for detect all", async () => {
  const html = await readFile("index.html", "utf8");

  assert.match(html, /id="batch-size"[^>]*value="1"/);
  assert.match(html, /id="batch-size"[^>]*min="1"[^>]*max="4"/);
  assert.match(html, /id="progress"/);
  assert.match(html, /id="summary"/);
  assert.match(html, /batchSizeInput/);
  assert.match(html, /Math\.min\(Math\.max\(Math\.trunc\(rawBatchSize\), 1\), 4\)/);
  assert.doesNotMatch(html, /max="8"/);
  assert.match(html, /detectLoadedPages/);
  assert.match(html, /progress\.value = processedPages/);
  assert.match(html, /progress\.max = pageCount/);
  assert.match(html, /Finished \$\{pageCount\} pages in/);
  assert.match(html, /\.legend\s*\{[\s\S]*?width:\s*132px/);
});

test("preview page runs wasm detection in a web worker", async () => {
  const html = await readFile("index.html", "utf8");

  assert.match(html, /new Worker\("wasm\/doclayout-worker\.js", \{ type: "module" \}\)/);
  assert.match(html, /requestWorker\("detectLoadedPage"/);
  assert.match(html, /requestWorker\("detectLoadedPages"/);
  assert.doesNotMatch(html, /new PPDocLayoutWasm\(\)/);
});

test("preview page yields frames between long worker requests", async () => {
  const html = await readFile("index.html", "utf8");

  assert.match(html, /function nextFrame\(\)/);
  assert.match(html, /const INTER_BATCH_PAUSE_MS = 60/);
  assert.match(html, /function sleep\(ms\)/);
  assert.match(html, /await nextFrame\(\);[\s\S]*?requestWorker\("detectLoadedPage"/);
  assert.match(html, /await nextFrame\(\);[\s\S]*?requestWorker\("detectLoadedPages"/);
  assert.match(html, /appendPage\(page\);[\s\S]*?await nextFrame\(\)/);
  assert.match(html, /await sleep\(INTER_BATCH_PAUSE_MS\)/);
});

test("preview page logs worker lifecycle and request timing", async () => {
  const html = await readFile("index.html", "utf8");

  assert.match(html, /const WORKER_LOG_PREFIX = "\[doclayout-worker\]"/);
  assert.match(html, /event\.data\?\.type === "worker-log"/);
  assert.match(html, /console\.info\(WORKER_LOG_PREFIX, event\.data\.message/);
  assert.match(html, /const startedAt = performance\.now\(\)/);
  assert.match(html, /pendingRequests\.set\(id, \{ resolve, reject, method, startedAt \}\)/);
  assert.match(html, /durationMs = Math\.round\(performance\.now\(\) - pending\.startedAt\)/);
  assert.match(html, /console\.info\(WORKER_LOG_PREFIX, "main request completed"/);
  assert.match(html, /console\.error\(WORKER_LOG_PREFIX, "worker error"/);
});

test("preview legend avoids horizontal scroll and can be dragged", async () => {
  const html = await readFile("index.html", "utf8");

  assert.match(html, /\.legend\s*\{[\s\S]*?overflow-x:\s*hidden/);
  assert.match(html, /\.legend-item span:last-child\s*\{[\s\S]*?overflow-wrap:\s*anywhere/);
  assert.match(html, /const legend = document\.querySelector\("#legend"\)/);
  assert.match(html, /legend\.addEventListener\("pointerdown"/);
  assert.match(html, /window\.addEventListener\("pointermove"/);
  assert.match(html, /legend\.style\.left = `\$\{nextLeft\}px`/);
  assert.match(html, /legend\.style\.top = `\$\{nextTop\}px`/);
});
