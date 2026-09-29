/**
 * Alpha MCP Layer — protocol shapes.
 *
 * Alpha implements the *shape* of the Model Context Protocol: tools are
 * described with a name, a description and a JSON Schema input, listed through
 * `tools/list` and invoked through `tools/call` over JSON-RPC 2.0. An external
 * MCP server can therefore be attached as a tool source without becoming the
 * centre of the system — Alpha's own model, permissions and audit trail stay in
 * charge of what happens.
 *
 * Nothing here contacts a server by itself: `HttpMcpClient` only runs when the
 * workspace configures a URL, and until then every MCP surface reports
 * NOT CONFIGURED.
 */

export type JsonRpcId = string | number | null;

export type JsonRpcRequest = {
  jsonrpc: "2.0";
  id: JsonRpcId;
  method: string;
  params?: unknown;
};

export type JsonRpcSuccess = {
  jsonrpc: "2.0";
  id: JsonRpcId;
  result: unknown;
};

export type JsonRpcFailure = {
  jsonrpc: "2.0";
  id: JsonRpcId;
  error: { code: number; message: string; data?: unknown };
};

export type JsonRpcResponse = JsonRpcSuccess | JsonRpcFailure;

export function isJsonRpcFailure(response: JsonRpcResponse): response is JsonRpcFailure {
  return "error" in response;
}

/** A tool as an MCP server describes it. */
export type McpToolDescriptor = {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  /** Some servers annotate risk; Alpha reads it when present. */
  annotations?: {
    title?: string;
    readOnlyHint?: boolean;
    destructiveHint?: boolean;
  };
};

export type McpServerInfo = {
  name: string;
  version: string;
  protocolVersion: string;
  capabilities?: Record<string, unknown>;
};

export type McpListToolsResult = {
  tools: McpToolDescriptor[];
};

export type McpContentBlock = {
  type: string;
  text?: string;
  data?: string;
  mimeType?: string;
};

export type McpCallToolResult = {
  content: McpContentBlock[];
  isError?: boolean;
};

export const MCP_PROTOCOL_VERSION = "2024-11-05";

export const MCP_METHODS = {
  initialize: "initialize",
  listTools: "tools/list",
  callTool: "tools/call",
} as const;

/** Flatten MCP content blocks into plain text for Alpha's own layers. */
export function mcpContentToText(content: McpContentBlock[]): string {
  return content
    .map((block) => {
      if (typeof block.text === "string") return block.text;
      if (typeof block.data === "string") return `[${block.type}:${block.mimeType ?? "binary"}]`;
      return `[${block.type}]`;
    })
    .join("\n");
}
