/**
 * Alpha Tools — schema validation.
 *
 * Alpha validates tool arguments against a JSON Schema subset. Using JSON
 * Schema as the canonical description (rather than a library-specific schema)
 * means the same descriptor can be shown in the workspace, checked by the tool
 * registry, and exchanged with an MCP server without translation loss.
 */

import { AlphaValidationError } from "../core/errors";

export type JsonSchemaType = "object" | "string" | "number" | "integer" | "boolean" | "array" | "null";

export type JsonSchema = {
  type?: JsonSchemaType | JsonSchemaType[];
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  enum?: (string | number | boolean)[];
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  minItems?: number;
  maxItems?: number;
  additionalProperties?: boolean;
  default?: unknown;
};

export type SchemaValidationResult = {
  ok: boolean;
  errors: string[];
};

function typeOf(value: unknown): JsonSchemaType {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  const t = typeof value;
  if (t === "string") return "string";
  if (t === "boolean") return "boolean";
  if (t === "number") return Number.isInteger(value) ? "integer" : "number";
  return "object";
}

function matchesType(value: unknown, expected: JsonSchemaType): boolean {
  const actual = typeOf(value);
  if (expected === "number") return actual === "number" || actual === "integer";
  if (expected === "integer") return actual === "integer" || (actual === "number" && Number.isInteger(value));
  return actual === expected;
}

function validateNode(value: unknown, schema: JsonSchema, path: string, errors: string[]): void {
  if (schema.type) {
    const expected = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!expected.some((type) => matchesType(value, type))) {
      errors.push(`${path}: expected ${expected.join(" | ")}, received ${typeOf(value)}`);
      return;
    }
  }
  if (schema.enum && !schema.enum.some((allowed) => allowed === value)) {
    errors.push(`${path}: value must be one of ${schema.enum.map(String).join(", ")}`);
  }
  if (typeof value === "string") {
    if (schema.minLength !== undefined && value.length < schema.minLength) {
      errors.push(`${path}: must be at least ${schema.minLength} characters`);
    }
    if (schema.maxLength !== undefined && value.length > schema.maxLength) {
      errors.push(`${path}: must be at most ${schema.maxLength} characters`);
    }
    if (schema.pattern && !new RegExp(schema.pattern, "u").test(value)) {
      errors.push(`${path}: must match /${schema.pattern}/`);
    }
  }
  if (typeof value === "number") {
    if (schema.minimum !== undefined && value < schema.minimum) {
      errors.push(`${path}: must be >= ${schema.minimum}`);
    }
    if (schema.maximum !== undefined && value > schema.maximum) {
      errors.push(`${path}: must be <= ${schema.maximum}`);
    }
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) {
      errors.push(`${path}: needs at least ${schema.minItems} items`);
    }
    if (schema.maxItems !== undefined && value.length > schema.maxItems) {
      errors.push(`${path}: accepts at most ${schema.maxItems} items`);
    }
    if (schema.items) {
      value.forEach((item, index) => validateNode(item, schema.items!, `${path}[${index}]`, errors));
    }
  }
  if (schema.type === "object" || (!schema.type && schema.properties)) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) return;
    const record = value as Record<string, unknown>;
    for (const key of schema.required ?? []) {
      if (record[key] === undefined) errors.push(`${path}.${key}: required`);
    }
    for (const [key, childSchema] of Object.entries(schema.properties ?? {})) {
      if (record[key] === undefined) continue;
      validateNode(record[key], childSchema, `${path}.${key}`, errors);
    }
    if (schema.additionalProperties === false) {
      const known = new Set(Object.keys(schema.properties ?? {}));
      for (const key of Object.keys(record)) {
        if (!known.has(key)) errors.push(`${path}.${key}: unexpected property`);
      }
    }
  }
}

export function validateAgainstSchema(value: unknown, schema: JsonSchema): SchemaValidationResult {
  const errors: string[] = [];
  validateNode(value, schema, "$", errors);
  return { ok: errors.length === 0, errors };
}

/** Validating variant that throws — used by the tool executor. */
export function assertAgainstSchema(value: unknown, schema: JsonSchema, toolName: string): void {
  const result = validateAgainstSchema(value, schema);
  if (!result.ok) {
    throw new AlphaValidationError(
      "tools",
      `arguments for "${toolName}" failed validation: ${result.errors.join("; ")}`,
      { tool: toolName, errors: result.errors },
    );
  }
}

/**
 * Build a JSON Schema object descriptor from a compact property map. Keeps tool
 * definitions short without losing the schema that callers and MCP clients see.
 */
export function objectSchema(
  properties: Record<string, JsonSchema>,
  required: string[] = [],
  description?: string,
): JsonSchema {
  return {
    type: "object",
    description,
    properties,
    required,
    additionalProperties: false,
  };
}
