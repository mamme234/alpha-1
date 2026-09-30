/**
 * Alpha Core — error taxonomy.
 *
 * Every subsystem throws one of these so callers (agents, workflows, the
 * observability layer) can branch on a failure class instead of matching
 * strings. Nothing here is a placeholder: these are the errors Alpha actually
 * raises at runtime.
 */

export type AlphaErrorModule =
  | "core"
  | "model"
  | "tokenizer"
  | "training"
  | "inference"
  | "embeddings"
  | "vector"
  | "rag"
  | "memory"
  | "agents"
  | "tools"
  | "mcp"
  | "automation"
  | "security"
  | "observability"
  | "datasets";

export class AlphaError extends Error {
  readonly code: string;
  readonly module: AlphaErrorModule;
  readonly details: Record<string, unknown>;

  constructor(
    code: string,
    module: AlphaErrorModule,
    message: string,
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "AlphaError";
    this.code = code;
    this.module = module;
    this.details = details;
  }
}

/** Input failed validation before reaching the model or a tool. */
export class AlphaValidationError extends AlphaError {
  constructor(module: AlphaErrorModule, message: string, details: Record<string, unknown> = {}) {
    super("alpha.validation", module, message, details);
    this.name = "AlphaValidationError";
  }
}

/** A caller tried something the policy engine does not permit. */
export class AlphaPermissionError extends AlphaError {
  constructor(module: AlphaErrorModule, message: string, details: Record<string, unknown> = {}) {
    super("alpha.permission_denied", module, message, details);
    this.name = "AlphaPermissionError";
  }
}

/** Rate limit or budget exhausted. */
export class AlphaRateLimitError extends AlphaError {
  constructor(module: AlphaErrorModule, message: string, details: Record<string, unknown> = {}) {
    super("alpha.rate_limited", module, message, details);
    this.name = "AlphaRateLimitError";
  }
}

/**
 * Raised when a caller asks for a capability that is architected but not yet
 * implemented — the honest alternative to returning a fabricated answer.
 */
export class AlphaNotImplementedError extends AlphaError {
  constructor(module: AlphaErrorModule, message: string, details: Record<string, unknown> = {}) {
    super("alpha.not_implemented", module, message, details);
    this.name = "AlphaNotImplementedError";
  }
}

/**
 * Raised when inference is attempted with a model that has no trained
 * parameters. Alpha can still run (the architecture is real), but callers must
 * acknowledge the model is untrained instead of presenting its output as
 * finished intelligence.
 */
export class AlphaUntrainedModelError extends AlphaError {
  constructor(message: string, details: Record<string, unknown> = {}) {
    super("alpha.untrained_model", "inference", message, details);
    this.name = "AlphaUntrainedModelError";
  }
}

/**
 * Raised when a checkpoint is corrupt, incomplete or was produced by a
 * different architecture. Alpha refuses to load it rather than running with
 * silently mismatched weights.
 */
export class AlphaCheckpointError extends AlphaError {
  readonly issues: string[];

  constructor(message: string, issues: string[] = [], details: Record<string, unknown> = {}) {
    super("alpha.checkpoint_invalid", "training", message, { ...details, issues });
    this.name = "AlphaCheckpointError";
    this.issues = issues;
  }
}

/** Raised when a tool or MCP endpoint fails during execution. */
export class AlphaToolError extends AlphaError {
  constructor(message: string, details: Record<string, unknown> = {}) {
    super("alpha.tool_failed", "tools", message, details);
    this.name = "AlphaToolError";
  }
}

export function isAlphaError(error: unknown): error is AlphaError {
  return error instanceof AlphaError;
}

/** Normalise anything thrown into a stable shape for logs and audit trails. */
export function describeError(error: unknown): { code: string; message: string; details: Record<string, unknown> } {
  if (isAlphaError(error)) {
    return { code: error.code, message: error.message, details: error.details };
  }
  if (error instanceof Error) {
    return { code: "alpha.unexpected", message: error.message, details: { name: error.name } };
  }
  return { code: "alpha.unexpected", message: String(error), details: {} };
}
