// list-checkpoints.ts — list every step-7 checkpoint chain step with loss, stage,
// and size, oldest first, so resumability and completion are auditable.
import { readdirSync, readFileSync } from "node:fs";

const dir = "src/alpha/experiments";
const files = readdirSync(dir)
  .filter((f) => f.endsWith(".alpha-ckpt.json"))
  .map((f) => {
    const m = f.match(/^step7-micro-v2-chunk-(\d+)\.alpha-ckpt\.json$/);
    if (!m) throw new Error(`unexpected checkpoint name ${f}`);
    return Number(m[1]);
  })
  .sort((a, b) => a - b);

console.log("checkpoint chain (oldest -> newest):");
for (const n of files) {
  const j = JSON.parse(readFileSync(`${dir}/step7-micro-v2-chunk-${n}.alpha-ckpt.json`, "utf8"));
  console.log(
    `  chunk ${String(n).padStart(2)}  step ${j.step}  trainLoss ${j.metrics.trainLoss.toFixed(4)}  stage ${j.stage}  size ${j.sizeBytes}`,
  );
}
const last = JSON.parse(readFileSync(`${dir}/step7-micro-v2-chunk-${files[files.length - 1]}.alpha-ckpt.json`, "utf8"));
console.log(`first trainLoss: ${files.map((n) => JSON.parse(readFileSync(`${dir}/step7-micro-v2-chunk-${n}.alpha-ckpt.json`, "utf8")).metrics.trainLoss).join(", ")}`);
console.log(`last trainLoss:  ${last.metrics.trainLoss.toFixed(4)}`);
console.log(`last step:       ${last.step}`);
