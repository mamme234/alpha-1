#!/usr/bin/env bun
// continue-step7.ts — repeatedly run the step-7 runner until step 64,
// using STEPS_PER_CHUNK=2 to stay within the host's memory ceiling.
// Dump a compact line per chunk, then a final summary.

function readStep(): number {
  const fs = require("fs");
  const dir = "src/alpha/experiments";
  let best = 0;
  if (!fs.existsSync(dir)) return 0;
  for (const f of fs.readdirSync(dir)) {
    const m = f.match(/^step7-micro-v2-chunk-(\d+)\.alpha-ckpt\.json$/);
    if (m) best = Math.max(best, Number(m[1]));
  }
  if (best === 0) return 0;
  const j = JSON.parse(fs.readFileSync(`${dir}/step7-micro-v2-chunk-${best}.alpha-ckpt.json`, "utf8"));
  return j.step;
}

function main(): void {
  const stepsPerChunk = parseInt(process.env.STEPS_PER_CHUNK ?? "2", 10);
  const total = 64;
  let iter = 0;
  const maxIters = parseInt(process.env.MAX_ITERS ?? "27", 10);

  while (true) {
    const stepBefore = readStep();
    console.log(`[continue] chunk n: step ${stepBefore} -> running (STEPS_PER_CHUNK=${stepsPerChunk})`);
    const { spawnSync } = require("child_process");
    const r = spawnSync("bun", ["scripts/alpha-train-step7.ts"], {
      stdio: "inherit",
      env: { ...process.env, STEPS_PER_CHUNK: String(stepsPerChunk) },
      cwd: "/home/daytona/codebase",
    });
    if (r.status !== 0) {
      console.error(`[continue] runner exited with status ${r.status}`);
      process.exit(r.status);
    }
    iter += 1;
    const stepAfter = readStep();
    console.log(`[continue] done: step now = ${stepAfter} (iter ${iter}/${maxIters})`);
    if (stepAfter >= total) {
      console.log(`[continue] target reached: step ${stepAfter} >= 64`);
      break;
    }
    if (iter >= maxIters) {
      console.log(`[continue] maxIters reached at step ${stepAfter}; stopping for user coordination.`);
      break;
    }
  }
  console.log(`[continue] FINAL step = ${readStep()}`);
}

main();
