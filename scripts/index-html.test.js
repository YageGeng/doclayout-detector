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
