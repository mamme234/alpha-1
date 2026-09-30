/**
 * Alpha CLI — run the real training lifecycle from a terminal.
 *
 *   bun scripts/alpha-train-smoke.ts
 *   bun scripts/alpha-train-smoke.ts --steps 120 --preset nano --verify
 *   bun scripts/alpha-train-smoke.ts --json > run.json
 *
 * This executes the genuine pipeline — corpus, tokenizer, transformer, loss,
 * backpropagation, AdamW, checkpoint, reload, resume, inference — and prints
 * exactly what happened. It writes nothing to disk, so it is safe to run
 * anywhere. Confidence in Alpha comes from running this, not from trusting a
 * number in a document.
 */

import type { AlphaModelPreset } from "../model/config";
import { ALPHA_SEED_CORPUS } from "../datasets/seed-corpus";
import { formatLifecycleReport, runAlphaTrainingLifecycle, type AlphaLifecycleReport } from "../training/lifecycle";
import { DEFAULT_LIFECYCLE_TRAINING } from "../training/lifecycle";

export type CliOptions = {
  preset: AlphaModelPreset;
  steps: number;
  batchSize: number;
  seqLen: number;
  seed: number;
  verify: boolean;
  json: boolean;
  prompt: string;
};

export function parseCliOptions(argv: string[]): CliOptions {
  const read = (flag: string): string | null => {
    const index = argv.indexOf(flag);
    return index >= 0 ? (argv[index + 1] ?? null) : null;
  };
  const preset = read("--preset");
  const options: CliOptions = {
    preset: preset === "micro" || preset === "small" || preset === "nano" ? preset : "nano",
    steps: Number(read("--steps") ?? DEFAULT_LIFECYCLE_TRAINING.totalSteps ?? 60),
    batchSize: Number(read("--batch") ?? DEFAULT_LIFECYCLE_TRAINING.batchSize ?? 8),
    seqLen: Number(read("--seq") ?? DEFAULT_LIFECYCLE_TRAINING.seqLen ?? 32),
    seed: Number(read("--seed") ?? DEFAULT_LIFECYCLE_TRAINING.seed ?? 1337),
    verify: argv.includes("--verify"),
    json: argv.includes("--json"),
    prompt: read("--prompt") ?? "Alpha is a self owned",
  };
  if (!Number.isFinite(options.steps) || options.steps < 1) throw new Error("--steps must be a positive integer");
  if (!Number.isFinite(options.batchSize) || options.batchSize < 1) throw new Error("--batch must be a positive integer");
  if (!Number.isFinite(options.seqLen) || options.seqLen < 2) throw new Error("--seq must be an integer of at least 2");
  if (!Number.isFinite(options.seed)) throw new Error("--seed must be a number");
  return options;
}

export function runLifecycleFromCli(options: CliOptions): AlphaLifecycleReport {
  return runAlphaTrainingLifecycle({
    preset: options.preset,
    dataset: ALPHA_SEED_CORPUS,
    training: {
      batchSize: options.batchSize,
      seqLen: options.seqLen,
      totalSteps: options.steps,
      warmupSteps: Math.max(1, Math.round(options.steps * 0.1)),
      evalInterval: Math.max(1, Math.round(options.steps / 4)),
      evalBatches: 4,
      checkpointInterval: Math.max(1, Math.round(options.steps / 2)),
      seed: options.seed,
      validationFraction: 0.12,
    },
    runVerification: options.verify,
    prompt: options.prompt,
  });
}

/** Print the report and return a process exit code. */
export function main(argv: string[] = [], write: (line: string) => void = console.log): number {
  let options: CliOptions;
  try {
    options = parseCliOptions(argv);
  } catch (error) {
    write(`alpha: ${error instanceof Error ? error.message : String(error)}`);
    return 2;
  }
  const report = runLifecycleFromCli(options);
  write(options.json ? JSON.stringify(report, null, 2) : formatLifecycleReport(report));
  return report.ok ? 0 : 1;
}
