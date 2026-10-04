// dump-ckpt.ts — dump the small fields of an on-disk alpha-ckpt.json
const path = Bun.argv[2];
const raw = await Bun.file(path).text();
let j: any;
try {
  j = JSON.parse(raw);
} catch (e) {
  console.error("INVALID JSON:", e);
  console.log(raw.slice(0, 500));
  process.exit(1);
}
const m = j.metrics;
console.log("file:", path);
console.log("step:", j.step);
console.log("stage:", j.stage);
console.log("isFineTune:", j.isFineTune);
console.log("metrics:", JSON.stringify(m));
console.log("has metrics trainLoss:", "trainLoss" in m, "trainLoss value:", j.metrics?.trainLoss);
console.log("summary checkpoint present:", j.summary?.checkpoint !== undefined);
