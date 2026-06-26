#!/usr/bin/env node

import {
  mkdir,
  open,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

export const defaultOptions = Object.freeze({
  endpoint: "https://huggingface.co",
  repo: "PaddlePaddle/PP-DocLayoutV3_safetensors",
  revision: "main",
  filename: "model.safetensors",
  output: join("models", "pp_doclayout_v3", "model.safetensors"),
  force: false,
});

/** Build the Hugging Face resolve URL for a model asset. */
export function buildModelUrl(endpoint, repo, filename, revision) {
  const base = endpoint.replace(/\/+$/, "");
  const encodedRepo = repo.split("/").map(encodeURIComponent).join("/");
  const encodedFilename = filename.split("/").map(encodeURIComponent).join("/");
  const encodedRevision = encodeURIComponent(revision);
  return `${base}/${encodedRepo}/resolve/${encodedRevision}/${encodedFilename}`;
}

/** Resolve environment defaults and CLI flags into one download configuration. */
export function resolveOptions(argv, env = process.env) {
  const options = {
    endpoint: env.HF_ENDPOINT ?? defaultOptions.endpoint,
    repo: env.PP_DOCLAYOUT_MODEL_REPO ?? defaultOptions.repo,
    revision: env.PP_DOCLAYOUT_MODEL_REVISION ?? defaultOptions.revision,
    filename: env.PP_DOCLAYOUT_MODEL_FILE ?? defaultOptions.filename,
    output: env.PP_DOCLAYOUT_MODEL_PATH ?? defaultOptions.output,
    force: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--force") {
      options.force = true;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }

    const value = argv[index + 1];
    if (!value) {
      throw new Error(`missing value for ${arg}`);
    }

    if (arg === "--endpoint") {
      options.endpoint = value;
    } else if (arg === "--repo") {
      options.repo = value;
    } else if (arg === "--revision") {
      options.revision = value;
    } else if (arg === "--filename") {
      options.filename = value;
    } else if (arg === "--output" || arg === "-o") {
      options.output = value;
    } else {
      throw new Error(`unknown option ${arg}`);
    }
    index += 1;
  }

  return options;
}

/** Validate that a file starts with a parseable safetensors metadata header. */
export async function ensureSafeTensorsFile(path) {
  const handle = await open(path, "r");
  try {
    const lengthBuffer = Buffer.alloc(8);
    const { bytesRead } = await handle.read(lengthBuffer, 0, lengthBuffer.length, 0);
    if (bytesRead !== lengthBuffer.length) {
      throw new Error("invalid safetensors header: missing metadata length");
    }

    const headerLength = Number(lengthBuffer.readBigUInt64LE());
    if (!Number.isSafeInteger(headerLength) || headerLength <= 0) {
      throw new Error("invalid safetensors header: invalid metadata length");
    }

    const fileStat = await handle.stat();
    if (headerLength + lengthBuffer.length > fileStat.size) {
      throw new Error("invalid safetensors header: metadata exceeds file size");
    }

    const header = Buffer.alloc(headerLength);
    await handle.read(header, 0, header.length, lengthBuffer.length);
    const metadata = JSON.parse(header.toString("utf8"));
    if (metadata === null || typeof metadata !== "object" || Array.isArray(metadata)) {
      throw new Error("invalid safetensors header: metadata is not an object");
    }
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error(`invalid safetensors header: ${error.message}`);
    }
    throw error;
  } finally {
    await handle.close();
  }
}

/** Download the configured model asset atomically and validate it before publishing. */
export async function downloadModel(options, env = process.env) {
  const output = options.output;
  const url = buildModelUrl(
    options.endpoint,
    options.repo,
    options.filename,
    options.revision,
  );

  if (!options.force && (await pathExists(output))) {
    await ensureSafeTensorsFile(output);
    console.log(`model already exists: ${output}`);
    return;
  }

  await mkdir(dirname(output), { recursive: true });
  const tempPath = `${output}.download-${process.pid}`;

  try {
    console.log(`downloading ${url}`);
    await downloadFile(url, tempPath, env);
    await ensureSafeTensorsFile(tempPath);
    await rename(tempPath, output);
    const { size } = await stat(output);
    console.log(`saved ${output} (${formatBytes(size)})`);
  } catch (error) {
    await rm(tempPath, { force: true });
    throw error;
  }
}

/** Fetch a URL to disk while reporting coarse progress for large model files. */
async function downloadFile(url, output, env) {
  const headers = {};
  if (env.HF_TOKEN) {
    headers.Authorization = `Bearer ${env.HF_TOKEN}`;
  }

  const response = await fetch(url, { headers, redirect: "follow" });
  if (!response.ok) {
    throw new Error(`download failed: HTTP ${response.status} ${response.statusText}`);
  }
  if (!response.body) {
    throw new Error("download failed: empty response body");
  }

  const total = Number(response.headers.get("content-length") ?? 0);
  const file = await open(output, "w");
  let downloaded = 0;
  let nextReport = 0;

  try {
    for await (const chunk of response.body) {
      await file.write(chunk);
      downloaded += chunk.length;

      // Large model downloads can be slow, so progress is intentionally coarse.
      if (total > 0 && downloaded >= nextReport) {
        console.error(`downloaded ${formatBytes(downloaded)} / ${formatBytes(total)}`);
        nextReport = downloaded + Math.max(Math.floor(total / 20), 1);
      }
    }
  } finally {
    await file.close();
  }
}

/** Return true when the path exists without treating other filesystem errors as hits. */
async function pathExists(path) {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

/** Format byte counts with stable one-decimal output for CLI messages. */
function formatBytes(bytes) {
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

/** Print CLI usage and the environment variables supported by the downloader. */
function printUsage() {
  console.log(`Usage: node scripts/download-model.js [options]

Downloads PP-DocLayoutV3 weights to:
  ${defaultOptions.output}

Options:
  --output, -o <path>      Output safetensors path
  --repo <repo-id>         Hugging Face repository id
  --revision <revision>    Repository revision, branch, or commit
  --filename <name>        Repository file to download
  --endpoint <url>         Hugging Face endpoint or mirror
  --force                  Re-download even when the output exists
  --help, -h               Show this help

Environment:
  HF_TOKEN                 Optional Hugging Face token
  HF_ENDPOINT              Optional Hugging Face endpoint or mirror
  PP_DOCLAYOUT_MODEL_PATH  Optional output path override`);
}

/** Run the downloader CLI and convert thrown errors into process exit codes. */
async function main() {
  const options = resolveOptions(process.argv.slice(2), process.env);
  if (options.help) {
    printUsage();
    return;
  }
  await downloadModel(options, process.env);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}
