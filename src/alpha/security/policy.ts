/**
 * Alpha Security — authentication context and authorization.
 *
 * Authentication is handled by the host application (Convex Auth in the
 * workspace). This module is the authorization half: every action carries an
 * actor id, every actor has a role, every role has permissions, and every
 * privileged operation calls `assert` before it runs.
 *
 * Two rules are enforced structurally rather than by convention:
 *   1. An agent never executes a tool outside its scope.
 *   2. A tool marked `requiresApproval` cannot run without an explicit,
 *      recorded human approval for that actor.
 */

import { AlphaPermissionError } from "../core/errors";

export type AlphaPermission =
  | "model.read"
  | "model.write"
  | "model.train"
  | "model.promote"
  | "inference.run"
  | "dataset.read"
  | "dataset.write"
  | "tokenizer.read"
  | "tokenizer.train"
  | "vector.read"
  | "vector.write"
  | "rag.ingest"
  | "rag.query"
  | "memory.read"
  | "memory.write"
  | "memory.write.long-term"
  | "memory.delete"
  | "tool.register"
  | "tool.execute"
  | "tool.execute.dangerous"
  | "mcp.connect"
  | "agent.run"
  | "agent.plan"
  | "workflow.read"
  | "workflow.write"
  | "workflow.run"
  | "observability.read"
  | "audit.read"
  | "policy.write";

export const ALL_PERMISSIONS: AlphaPermission[] = [
  "model.read",
  "model.write",
  "model.train",
  "model.promote",
  "inference.run",
  "dataset.read",
  "dataset.write",
  "tokenizer.read",
  "tokenizer.train",
  "vector.read",
  "vector.write",
  "rag.ingest",
  "rag.query",
  "memory.read",
  "memory.write",
  "memory.write.long-term",
  "memory.delete",
  "tool.register",
  "tool.execute",
  "tool.execute.dangerous",
  "mcp.connect",
  "agent.run",
  "agent.plan",
  "workflow.read",
  "workflow.write",
  "workflow.run",
  "observability.read",
  "audit.read",
  "policy.write",
];

export type AlphaRole = "owner" | "operator" | "member" | "viewer" | "agent" | "system";

export const DEFAULT_ROLE_PERMISSIONS: Record<AlphaRole, AlphaPermission[]> = {
  owner: [...ALL_PERMISSIONS],
  operator: [
    "model.read",
    "model.train",
    "inference.run",
    "dataset.read",
    "dataset.write",
    "tokenizer.read",
    "tokenizer.train",
    "vector.read",
    "vector.write",
    "rag.ingest",
    "rag.query",
    "memory.read",
    "memory.write",
    "memory.delete",
    "tool.execute",
    "agent.run",
    "agent.plan",
    "workflow.read",
    "workflow.write",
    "workflow.run",
    "observability.read",
    "audit.read",
  ],
  member: [
    "model.read",
    "inference.run",
    "dataset.read",
    "tokenizer.read",
    "vector.read",
    "rag.query",
    "memory.read",
    "memory.write",
    "memory.delete",
    "tool.execute",
    "agent.run",
    "workflow.read",
    "workflow.run",
    "observability.read",
  ],
  viewer: ["model.read", "inference.run", "vector.read", "rag.query", "observability.read"],
  // Agents start with the minimum: read-only retrieval and inference, no writes.
  agent: ["model.read", "inference.run", "vector.read", "rag.query", "memory.read", "tool.execute"],
  system: [...ALL_PERMISSIONS],
};

export type PolicyDecision = {
  allowed: boolean;
  actorId: string;
  permission: AlphaPermission;
  reason: string;
  /** Approval token that unblocked a gated action, when applicable. */
  approvalId?: string;
};

export type AgentScope = {
  agentId: string;
  /** Permissions the agent may exercise, intersected with its role. */
  permissions: AlphaPermission[];
  /** Tool names the agent may call. An empty list means "no tools". */
  allowedTools: string[];
  maxSteps: number;
};

export type GatedToolDescriptor = {
  name: string;
  permission: AlphaPermission;
  /** Human approval required before each execution. */
  requiresApproval?: boolean;
};

type Approval = {
  id: string;
  actorId: string;
  toolName: string;
  grantedAt: number;
  grantedBy: string;
  expiresAt: number;
};

export type PolicyEvent = {
  at: number;
  actorId: string;
  permission: AlphaPermission;
  allowed: boolean;
  reason: string;
  resource?: string;
};

export class AlphaPolicyEngine {
  private roleAssignments = new Map<string, AlphaRole>();
  private scopes = new Map<string, AgentScope>();
  private approvals = new Map<string, Approval>();
  private events: PolicyEvent[] = [];
  private readonly approvalTtlMs: number;

  constructor(options: { approvalTtlMs?: number } = {}) {
    this.approvalTtlMs = options.approvalTtlMs ?? 5 * 60_000;
  }

  assignRole(actorId: string, role: AlphaRole): void {
    this.roleAssignments.set(actorId, role);
  }

  roleOf(actorId: string): AlphaRole | null {
    return this.roleAssignments.get(actorId) ?? null;
  }

  /** Effective permissions = role permissions, narrowed by an agent scope. */
  permissionsFor(actorId: string): AlphaPermission[] {
    const role = this.roleOf(actorId);
    if (!role) return [];
    const base = DEFAULT_ROLE_PERMISSIONS[role];
    const scope = this.scopes.get(actorId);
    if (!scope) return base;
    return base.filter((permission) => scope.permissions.includes(permission));
  }

  registerAgentScope(scope: AgentScope): void {
    const allowed = scope.permissions.filter((permission) => ALL_PERMISSIONS.includes(permission));
    this.scopes.set(scope.agentId, { ...scope, permissions: allowed });
  }

  scopeOf(agentId: string): AgentScope | null {
    return this.scopes.get(agentId) ?? null;
  }

  /** Human approval for one gated tool, valid for a short window. */
  approveTool(input: { actorId: string; toolName: string; grantedBy: string }): Approval {
    const approval: Approval = {
      id: `approval_${input.actorId}_${input.toolName}_${Date.now().toString(36)}`,
      actorId: input.actorId,
      toolName: input.toolName,
      grantedAt: Date.now(),
      grantedBy: input.grantedBy,
      expiresAt: Date.now() + this.approvalTtlMs,
    };
    this.approvals.set(approval.id, approval);
    return approval;
  }

  revokeApproval(id: string): boolean {
    return this.approvals.delete(id);
  }

  private findApproval(actorId: string, toolName: string): Approval | null {
    const now = Date.now();
    for (const approval of this.approvals.values()) {
      if (approval.actorId !== actorId || approval.toolName !== toolName) continue;
      if (approval.expiresAt < now) continue;
      return approval;
    }
    return null;
  }

  /** Non-throwing check. */
  check(actorId: string, permission: AlphaPermission, resource?: string): PolicyDecision {
    const role = this.roleOf(actorId);
    if (!role) {
      const decision: PolicyDecision = {
        allowed: false,
        actorId,
        permission,
        reason: `actor "${actorId}" has no role assigned`,
      };
      this.record(decision, resource);
      return decision;
    }
    const permissions = this.permissionsFor(actorId);
    const allowed = permissions.includes(permission);
    const decision: PolicyDecision = {
      allowed,
      actorId,
      permission,
      reason: allowed
        ? `role "${role}" grants "${permission}"`
        : `role "${role}" does not grant "${permission}"${this.scopes.has(actorId) ? " within this agent scope" : ""}`,
    };
    this.record(decision, resource);
    return decision;
  }

  /** Throwing check — the form used by tools, agents and workflows. */
  assert(actorId: string, permission: AlphaPermission, resource?: string): void {
    const decision = this.check(actorId, permission, resource);
    if (!decision.allowed) {
      throw new AlphaPermissionError("security", decision.reason, {
        actorId,
        permission,
        resource,
      });
    }
  }

  /**
   * Gate for tool execution: permission check, agent tool allow-list, and the
   * approval requirement for dangerous tools. This is the single choke point
   * every tool call in Alpha passes through.
   */
  checkTool(
    actorId: string,
    tool: GatedToolDescriptor,
    resource?: string,
  ): PolicyDecision {
    const scope = this.scopes.get(actorId);
    if (scope && scope.allowedTools.length > 0 && !scope.allowedTools.includes(tool.name)) {
      const decision: PolicyDecision = {
        allowed: false,
        actorId,
        permission: tool.permission,
        reason: `tool "${tool.name}" is outside agent scope for "${actorId}"`,
      };
      this.record(decision, resource ?? tool.name);
      return decision;
    }
    if (scope && scope.allowedTools.length === 0) {
      const decision: PolicyDecision = {
        allowed: false,
        actorId,
        permission: tool.permission,
        reason: `agent "${actorId}" has no tools enabled`,
      };
      this.record(decision, resource ?? tool.name);
      return decision;
    }
    const base = this.check(actorId, tool.permission, resource ?? tool.name);
    if (!base.allowed) return base;
    if (tool.requiresApproval) {
      const approval = this.findApproval(actorId, tool.name);
      if (!approval) {
        const decision: PolicyDecision = {
          allowed: false,
          actorId,
          permission: tool.permission,
          reason: `tool "${tool.name}" requires explicit approval, which has not been granted`,
        };
        this.record(decision, resource ?? tool.name);
        return decision;
      }
      return { ...base, approvalId: approval.id, reason: `${base.reason}; approved via ${approval.id}` };
    }
    return base;
  }

  assertTool(actorId: string, tool: GatedToolDescriptor, resource?: string): PolicyDecision {
    const decision = this.checkTool(actorId, tool, resource);
    if (!decision.allowed) {
      throw new AlphaPermissionError("security", decision.reason, {
        actorId,
        tool: tool.name,
        permission: tool.permission,
      });
    }
    return decision;
  }

  /** Recent decisions, newest first — surfaced in the observability panel. */
  recentEvents(limit = 50): PolicyEvent[] {
    return this.events.slice(-limit).reverse();
  }

  private record(decision: PolicyDecision, resource?: string): void {
    this.events.push({
      at: Date.now(),
      actorId: decision.actorId,
      permission: decision.permission,
      allowed: decision.allowed,
      reason: decision.reason,
      resource,
    });
    if (this.events.length > 1000) this.events = this.events.slice(-500);
  }
}

/** Permission a tool needs, inferred from its declared dangerous flag. */
export function permissionForTool(input: {
  dangerous?: boolean;
  permission?: AlphaPermission;
}): AlphaPermission {
  if (input.permission) return input.permission;
  return input.dangerous ? "tool.execute.dangerous" : "tool.execute";
}
