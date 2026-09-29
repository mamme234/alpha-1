/**
 * Alpha MCP Layer — client and tool adapter.
 *
 * `HttpMcpClient` speaks real JSON-RPC 2.0 over HTTP to a configured MCP
 * endpoint. `registerMcpTools` converts a server's tool descriptors into Alpha
 * tools, so MCP tools inherit Alpha's argument validation, permission checks,
 * rate limits, sandbox rules and audit trail automatically.
 */

import { alphaId } from "../core/types";
import { AlphaToolError, AlphaValidationError } from "../core/errors";
import type { AlphaPermission } from "../security/policy";
import type { AlphaToolDefinition, AlphaToolRegistry, ToolDescriptor } from "../tools/registry";
import { objectSchema, type JsonSchema } from "../tools/schema";
import {
  MCP_METHODS,
  MCP_PROTOCOL_VERSION,
  type JsonRpcResponse,
  type McpCallToolResult,
  type McpListToolsResult,
  type McpServerInfo,
  type McpToolDescriptor,
  isJsonRpcFailure,
  mcpContentToText,
} from "./protocol";

export type McpClientInfo = {
  connected: boolean;
  transport: "http" | "none";
  endpoint: string | null;
  server: McpServerInfo | null;
  error: string | null;
};

export interface McpClient {
  readonly info: McpClientInfo;
  connect(): Promise<McpServerInfo>;
  listTools(): Promise<McpToolDescriptor[]>;
  callTool(name: string, args: Record<string, unknown>): Promise<McpCallToolResult>;
}

export type HttpMcpClientOptions = {
  endpoint: string;
  /** Sent as the Authorization header when set. Never logged. */
  bearerToken?: string;
  headers?: Record<string, string>;
  timeoutMs?: number;
  clientName?: string;
};

/**
 * Minimal MCP client over HTTP. It performs a real `initialize` handshake and
 * real `tools/list` / `tools/call` requests; it is "not configured" only in the
 * sense that Alpha ships without an endpoint, because there is no MCP server to
 * point at until a user runs one.
 */
export class HttpMcpClient implements McpClient {
  private readonly endpoint: string;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;
  private readonly clientName: string;
  private serverInfo: McpServerInfo | null = null;
  private lastError: string | null = null;
  private connected = false;
  private requestCounter = 0;
  /** Tool descriptors cached from the most recent successful `tools/list`. */
  private cachedTools: McpToolDescriptor[] = [];

  constructor(options: HttpMcpClientOptions) {
    if (!/^https?:\/\//.test(options.endpoint)) {
      throw new AlphaValidationError("mcp", "MCP endpoint must be an http(s) URL");
    }
    this.endpoint = options.endpoint;
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.clientName = options.clientName ?? "alpha";
    this.headers = {
      "content-type": "application/json",
      accept: "application/json",
      ...(options.bearerToken ? { authorization: `Bearer ${options.bearerToken}` } : {}),
      ...(options.headers ?? {}),
    };
  }

  get info(): McpClientInfo {
    return {
      connected: this.connected,
      transport: "http",
      endpoint: this.endpoint,
      server: this.serverInfo,
      error: this.lastError,
    };
  }

  get tools(): McpToolDescriptor[] {
    return [...this.cachedTools];
  }

  private async request(method: string, params?: unknown): Promise<unknown> {
    const id = ++this.requestCounter;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(this.endpoint, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new AlphaToolError(`MCP endpoint returned HTTP ${response.status}`, {
          endpoint: this.endpoint,
          method,
        });
      }
      const payload = (await response.json()) as JsonRpcResponse;
      if (isJsonRpcFailure(payload)) {
        throw new AlphaToolError(`MCP error ${payload.error.code}: ${payload.error.message}`, {
          method,
        });
      }
      this.lastError = null;
      return payload.result;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.lastError = message;
      this.connected = false;
      throw error instanceof AlphaToolError
        ? error
        : new AlphaToolError(`MCP request failed: ${message}`, { method, endpoint: this.endpoint });
    } finally {
      clearTimeout(timer);
    }
  }

  async connect(): Promise<McpServerInfo> {
    const result = (await this.request(MCP_METHODS.initialize, {
      protocolVersion: MCP_PROTOCOL_VERSION,
      clientInfo: { name: this.clientName, version: "1.0.0" },
      capabilities: { tools: {} },
    })) as Partial<McpServerInfo> | null;
    this.serverInfo = {
      name: result?.name ?? "unknown-mcp-server",
      version: result?.version ?? "unknown",
      protocolVersion: result?.protocolVersion ?? MCP_PROTOCOL_VERSION,
      capabilities: result?.capabilities,
    };
    this.connected = true;
    return this.serverInfo;
  }

  async listTools(): Promise<McpToolDescriptor[]> {
    const result = (await this.request(MCP_METHODS.listTools, {})) as McpListToolsResult | null;
    this.cachedTools = Array.isArray(result?.tools) ? result!.tools : [];
    return this.cachedTools;
  }

  async callTool(name: string, args: Record<string, unknown>): Promise<McpCallToolResult> {
    const result = (await this.request(MCP_METHODS.callTool, { name, arguments: args })) as
      | McpCallToolResult
      | null;
    if (!result || !Array.isArray(result.content)) {
      throw new AlphaToolError(`MCP tool "${name}" returned an unexpected payload`, { tool: name });
    }
    return result;
  }
}

/** Convert a JSON Schema fragment from an MCP server into Alpha's schema type. */
export function convertJsonSchema(schema: Record<string, unknown> | undefined): JsonSchema {
  if (!schema || typeof schema !== "object") {
    return objectSchema({}, []);
  }
  const type = typeof schema.type === "string" ? (schema.type as JsonSchema["type"]) : undefined;
  const properties: Record<string, JsonSchema> = {};
  const rawProperties = schema.properties;
  if (rawProperties && typeof rawProperties === "object") {
    for (const [key, value] of Object.entries(rawProperties as Record<string, unknown>)) {
      properties[key] = convertJsonSchema(value as Record<string, unknown>);
    }
  }
  const required = Array.isArray(schema.required) ? (schema.required as string[]) : [];
  const converted: JsonSchema = {
    type,
    description: typeof schema.description === "string" ? schema.description : undefined,
    properties: Object.keys(properties).length ? properties : undefined,
    required: required.length ? required : undefined,
    enum: Array.isArray(schema.enum) ? (schema.enum as (string | number | boolean)[]) : undefined,
    minimum: typeof schema.minimum === "number" ? schema.minimum : undefined,
    maximum: typeof schema.maximum === "number" ? schema.maximum : undefined,
    minLength: typeof schema.minLength === "number" ? schema.minLength : undefined,
    maxLength: typeof schema.maxLength === "number" ? schema.maxLength : undefined,
    pattern: typeof schema.pattern === "string" ? schema.pattern : undefined,
  };
  if (schema.items && typeof schema.items === "object") {
    converted.items = convertJsonSchema(schema.items as Record<string, unknown>);
  }
  return converted;
}

export type RegisterMcpToolsOptions = {
  /** Permission Alpha requires before calling these tools. */
  permission?: AlphaPermission;
  /** Grant human approval per call. Defaults to the server's destructive hint. */
  requiresApproval?: boolean;
  /** Tools to skip (useful for denying specific server capabilities). */
  deny?: string[];
  /** Prefix applied to tool names registered in Alpha. */
  prefix?: string;
};

/**
 * Register every tool a server advertises as an Alpha tool.
 *
 * The handler performs a real `tools/call` and flattens the MCP content blocks
 * into text. Failures surface as `AlphaToolError` and are recorded by the
 * registry like any other tool failure.
 */
export async function registerMcpTools(
  registry: AlphaToolRegistry,
  client: McpClient,
  options: RegisterMcpToolsOptions = {},
): Promise<ToolDescriptor[]> {
  const serverTools = await client.listTools();
  const prefix = options.prefix ?? "mcp";
  const deny = new Set(options.deny ?? []);
  const registered: ToolDescriptor[] = [];
  for (const tool of serverTools) {
    if (deny.has(tool.name)) continue;
    const alphaName = `${prefix}.${tool.name}`.replace(/[^a-z0-9._]/gi, "_").toLowerCase();
    if (registry.has(alphaName)) continue;
    const destructive = tool.annotations?.destructiveHint === true;
    const definition: AlphaToolDefinition<Record<string, unknown>, Record<string, unknown>> = {
      name: alphaName,
      description: `${tool.description ?? "MCP tool"} (provided by an MCP server; arguments validated by Alpha before the call).`,
      module: "mcp",
      version: "0.1.0",
      inputSchema: convertJsonSchema(tool.inputSchema),
      permission: options.permission ?? (destructive ? "tool.execute.dangerous" : "tool.execute"),
      source: "mcp",
      requiresApproval: options.requiresApproval ?? destructive,
      dangerous: destructive,
      characteristics: { networked: true, mutates: destructive },
      tags: ["mcp", "external", "tools"],
      handler: async (args) => {
        const result = await client.callTool(tool.name, args);
        return {
          ok: result.isError !== true,
          text: mcpContentToText(result.content),
          blocks: result.content.length,
          server: client.info.server?.name ?? "unknown",
        };
      },
      verify: (output) => ({
        ok: output.ok !== false,
        reason: output.ok === false ? "MCP server reported an error" : "MCP tool returned content",
      }),
    };
    registered.push(registry.register(definition));
  }
  return registered;
}

/** A client that is deliberately not connected — used to show NOT CONFIGURED. */
export class UnconfiguredMcpClient implements McpClient {
  private readonly reason: string;

  constructor(reason = "no MCP endpoint configured") {
    this.reason = reason;
  }

  get info(): McpClientInfo {
    return {
      connected: false,
      transport: "none",
      endpoint: null,
      server: null,
      error: this.reason,
    };
  }

  async connect(): Promise<McpServerInfo> {
    throw new AlphaToolError(`MCP is not configured: ${this.reason}`, { mcp: "not-configured" });
  }

  async listTools(): Promise<McpToolDescriptor[]> {
    return [];
  }

  async callTool(name: string): Promise<McpCallToolResult> {
    throw new AlphaToolError(`MCP is not configured: ${this.reason}`, { tool: name });
  }
}

/** Stable id helper reused when MCP tools need to correlate calls. */
export function mcpCallId(): string {
  return alphaId("mcp");
}
