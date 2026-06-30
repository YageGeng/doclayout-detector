import init, { PPDocLayoutWasm } from "../pkg/doclayout_detector.js";

let detector = null;
let bootPromise = null;

async function bootDetector() {
  if (detector) return detector;
  if (!bootPromise) {
    const startedAt = performance.now();
    postWorkerLog("worker boot started");
    bootPromise = init().then(() => {
      detector = new PPDocLayoutWasm();
      postWorkerLog("worker boot completed", {
        durationMs: Math.round(performance.now() - startedAt),
      });
      return detector;
    }).catch((error) => {
      bootPromise = null;
      postWorkerLog("worker boot failed", {
        durationMs: Math.round(performance.now() - startedAt),
        error: error instanceof Error ? error.message : String(error),
      });
      throw error;
    });
  }
  return bootPromise;
}

function postWorkerLog(message, data = {}) {
  self.postMessage({
    type: "worker-log",
    level: "info",
    message,
    data: {
      timestampMs: Math.round(performance.now()),
      ...data,
    },
  });
}

function transferablesForResult(result) {
  const transferables = [];
  const collectPage = (page) => {
    const imageBytes = page?.imageBytes;
    if (ArrayBuffer.isView(imageBytes)) {
      transferables.push(imageBytes.buffer);
    } else if (imageBytes instanceof ArrayBuffer) {
      transferables.push(imageBytes);
    }
  };

  if (Array.isArray(result)) {
    for (const page of result) {
      collectPage(page);
    }
  } else {
    collectPage(result);
  }

  return transferables;
}

self.addEventListener("message", async (event) => {
  const { id, method, params } = event.data;
  const startedAt = performance.now();
  postWorkerLog("worker request started", { id, method });

  try {
    const detector = await bootDetector();
    let result;
    switch (method) {
      case "boot":
        result = true;
        break;
      case "loadPdf":
        result = detector.loadPdf(new Uint8Array(params.bytes));
        break;
      case "detectLoadedPage":
        result = await detector.detectLoadedPage(params.pageNumber, params.dpi);
        break;
      case "detectLoadedPages":
        result = await detector.detectLoadedPages(params.startPage, params.count, params.dpi);
        break;
      default:
        throw new Error(`Unknown worker method: ${method}`);
    }
    const transferables = transferablesForResult(result);
    postWorkerLog("worker request completed", {
      id,
      method,
      durationMs: Math.round(performance.now() - startedAt),
      transferables: transferables.length,
    });
    self.postMessage({ id, ok: true, result }, transferables);
  } catch (error) {
    postWorkerLog("worker request failed", {
      id,
      method,
      durationMs: Math.round(performance.now() - startedAt),
      error: error instanceof Error ? error.message : String(error),
    });
    self.postMessage({
      id,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
