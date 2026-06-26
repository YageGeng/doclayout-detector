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
