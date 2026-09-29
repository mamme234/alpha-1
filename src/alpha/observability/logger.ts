/**
 * Alpha Observability — structured logging.
 *
 * Log entries are objects, not formatted strings, so they can be searched,
 * rendered in the workspace, and persisted without re-parsing. Anything that
 * looks like a secret is redacted before the entry is stored, because an audit
 * trail that leaks keys is worse than no audit trail.
 */

import { redactSecrets } from "../security/validation";

export type LogLevel = "debug" | "info" | "warn" | "error";

export type LogEntry = {
  id: number;
  at: number;
  level: LogLevel;
  module: string;
  message: string;
  traceId: string | null;
  data: Record<string, unknown>;
};

const LEVEL_ORDER: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };

export class AlphaLogger {
  private entries: LogEntry[] = [];
  private counter = 0;
  private readonly maxEntries: number;
  private minLevel: LogLevel;
  private readonly listeners = new Set<(entry: LogEntry) => void>();

  constructor(options: { maxEntries?: number; minLevel?: LogLevel } = {}) {
    this.maxEntries = options.maxEntries ?? 1000;
    this.minLevel = options.minLevel ?? "info";
  }

  setLevel(level: LogLevel): void {
    this.minLevel = level;
  }

  child(module: string, traceId: string | null = null): ModuleLogger {
    return new ModuleLogger(this, module, traceId);
  }

  log(level: LogLevel, module: string, message: string, data: Record<string, unknown> = {}, traceId: string | null = null): LogEntry | null {
    if (LEVEL_ORDER[level] < LEVEL_ORDER[this.minLevel]) return null;
    const entry: LogEntry = {
      id: ++this.counter,
      at: Date.now(),
      level,
      module,
      message: redactSecrets(message),
      traceId,
      data: safeData(data),
    };
    this.entries.push(entry);
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(-Math.floor(this.maxEntries * 0.75));
    }
    for (const listener of this.listeners) listener(entry);
    return entry;
  }

  onEntry(listener: (entry: LogEntry) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  list(options: { limit?: number; level?: LogLevel; module?: string } = {}): LogEntry[] {
    return this.entries
      .filter((entry) =>
        options.level ? LEVEL_ORDER[entry.level] >= LEVEL_ORDER[options.level] : true,
      )
      .filter((entry) => (options.module ? entry.module === options.module : true))
      .slice(-(options.limit ?? 100))
      .reverse();
  }

  counts(): Record<LogLevel, number> {
    const counts: Record<LogLevel, number> = { debug: 0, info: 0, warn: 0, error: 0 };
    for (const entry of this.entries) counts[entry.level]++;
    return counts;
  }

  /** JSON-lines export, suitable for a file or a log pipeline. */
  toJsonLines(limit = 200): string {
    return this.list({ limit })
      .reverse()
      .map((entry) => JSON.stringify(entry))
      .join("\n");
  }

  clear(): void {
    this.entries = [];
  }
}

function safeData(data: Record<string, unknown>): Record<string, unknown> {
  try {
    return JSON.parse(redactSecrets(JSON.stringify(data))) as Record<string, unknown>;
  } catch {
    return { note: "data was not serialisable", keys: Object.keys(data) };
  }
}

/** Logger bound to a module and trace, so callers do not repeat themselves. */
export class ModuleLogger {
  private readonly logger: AlphaLogger;
  readonly module: string;
  readonly traceId: string | null;

  constructor(logger: AlphaLogger, module: string, traceId: string | null = null) {
    this.logger = logger;
    this.module = module;
    this.traceId = traceId;
  }

  debug(message: string, data?: Record<string, unknown>): void {
    this.logger.log("debug", this.module, message, data, this.traceId);
  }

  info(message: string, data?: Record<string, unknown>): void {
    this.logger.log("info", this.module, message, data, this.traceId);
  }

  warn(message: string, data?: Record<string, unknown>): void {
    this.logger.log("warn", this.module, message, data, this.traceId);
  }

  error(message: string, data?: Record<string, unknown>): void {
    this.logger.log("error", this.module, message, data, this.traceId);
  }
}
