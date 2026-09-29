/**
 * Alpha Core — shared vocabulary for lifecycle status and identifiers.
 *
 * Honesty rule for the whole stack: a component's status is derived from what
 * actually exists (code, weights, checkpoints), never from what we wish were
 * true. These are the only labels Alpha is allowed to print about itself.
 */

/**
 * Lifecycle label for an Alpha module or artifact.
 *
 * - `planned`        — architecture documented, no runnable code yet
 * - `in-development` — code exists but has known gaps
 * - `untrained`      — runnable, parameters are random initialisation
 * - `not-configured` — implemented, needs configuration (endpoint, keys, data)
 * - `ready`          — implemented and verified by tests or a completed run
 */
export type AlphaStatus = "planned" | "in-development" | "untrained" | "not-configured" | "ready";

/**
 * The five model states Alpha must never conflate.
 * `architecture` is the code shape; the rest describe the weights.
 */
export type AlphaModelStage =
  | "architecture"
  | "untrained"
  | "trained"
  | "fine-tuned"
  | "production";

export const ALPHA_STATUS_ORDER: AlphaStatus[] = [
  "planned",
  "in-development",
  "untrained",
  "not-configured",
  "ready",
];

export function statusLabel(status: AlphaStatus): string {
  switch (status) {
    case "planned":
      return "PLANNED";
    case "in-development":
      return "IN DEVELOPMENT";
    case "untrained":
      return "UNTRAINED";
    case "not-configured":
      return "NOT CONFIGURED";
    case "ready":
      return "READY";
  }
}

export function modelStageLabel(stage: AlphaModelStage): string {
  switch (stage) {
    case "architecture":
      return "ARCHITECTURE ONLY";
    case "untrained":
      return "UNTRAINED";
    case "trained":
      return "TRAINED (FROM SCRATCH)";
    case "fine-tuned":
      return "FINE-TUNED";
    case "production":
      return "PRODUCTION";
  }
}

export type AlphaModuleDescriptor = {
  /** Directory name inside `src/alpha`. */
  id: string;
  name: string;
  summary: string;
  status: AlphaStatus;
  /** Implementation notes shown in the workspace, kept honest on purpose. */
  notes: string[];
};

/** Monotonic-ish unique id, prefixed so ids are readable in logs. */
export function alphaId(prefix: string): string {
  const time = Date.now().toString(36);
  const rand = Math.floor(Math.random() * 0xffffff).toString(36).padStart(5, "0");
  return `${prefix}_${time}${rand}`;
}

/** 128-bit-ish trace id in hex, used to correlate spans across modules. */
export function newTraceId(): string {
  const chunks: string[] = [];
  for (let i = 0; i < 4; i++) {
    chunks.push(Math.floor(Math.random() * 0xffffffff).toString(16).padStart(8, "0"));
  }
  return chunks.join("");
}

export function newSpanId(): string {
  return Math.floor(Math.random() * 0xffffffff).toString(16).padStart(8, "0");
}

export function byteLength(text: string): number {
  return new TextEncoder().encode(text).length;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function round(value: number, digits = 4): number {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}
