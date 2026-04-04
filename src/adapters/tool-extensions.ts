/**
 * tool-extensions.ts — Swiss-army-tool integration extensions for oboto-agent.
 *
 * This module provides:
 * 1. AgentDynamicTools  — A DynamicBranchNode subclass for runtime tool discovery
 * 2. createToolTimingMiddleware — Swiss-army-tool Middleware that emits timing events
 * 3. createAgentToolTree — Helper to build a pre-wired tool tree with memory + dynamic tools
 */

import {
  DynamicBranchNode,
  BranchNode,
  LeafNode,
  TreeBuilder,
  SessionManager,
  createMemoryModule,
  type Middleware,
  type ExecutionContext,
} from "@sschepis/swiss-army-tool";
import type { AgentEventBus } from "../event-bus.js";

// ── Types ─────────────────────────────────────────────────────────────

export interface DynamicToolProvider {
  /**
   * Called when the branch needs refreshing.
   * Return an array of leaf node configs to populate the dynamic branch.
   */
  discover(): Promise<DynamicToolEntry[]> | DynamicToolEntry[];
}

export interface DynamicToolEntry {
  name: string;
  description: string;
  requiredArgs?: string[] | Record<string, { type?: string; description?: string }>;
  optionalArgs?: string[] | Record<string, { type?: string; description?: string; default?: unknown }>;
  handler: (kwargs: Record<string, unknown>) => string | Promise<string>;
}

export interface AgentToolTreeConfig {
  /** Session manager for the swiss-army-tool Router */
  session: SessionManager;
  /** Whether to include the memory module (set/get/list/pin). Default: true */
  includeMemory?: boolean;
  /** Additional static branches to add to the tree */
  staticBranches?: BranchNode[];
  /** Dynamic tool providers to register. Each entry becomes a DynamicBranchNode */
  dynamicProviders?: Array<{
    name: string;
    description: string;
    provider: DynamicToolProvider;
    /** TTL in ms before the provider is re-queried. Default: 60000 */
    ttlMs?: number;
  }>;
}

// ── AgentDynamicTools ─────────────────────────────────────────────────

/**
 * A DynamicBranchNode that discovers tools at runtime from a provider.
 *
 * Use this to integrate external tool sources (MCP servers, plugin systems,
 * API discovery endpoints, etc.) into the swiss-army-tool command tree.
 *
 * The provider's `discover()` method is called when the TTL expires,
 * and the returned entries are registered as LeafNode children.
 *
 * ```ts
 * const mcpTools = new AgentDynamicTools({
 *   name: "mcp",
 *   description: "Tools discovered from MCP servers",
 *   provider: myMcpProvider,
 *   ttlMs: 30_000, // refresh every 30s
 * });
 * ```
 */
export class AgentDynamicTools extends DynamicBranchNode {
  private provider: DynamicToolProvider;

  constructor(config: {
    name: string;
    description: string;
    provider: DynamicToolProvider;
    ttlMs?: number;
  }) {
    super({
      name: config.name,
      description: config.description,
      ttlMs: config.ttlMs ?? 60_000,
    });
    this.provider = config.provider;
  }

  protected async refresh(): Promise<void> {
    const entries = await this.provider.discover();
    for (const entry of entries) {
      this.addChild(
        new LeafNode({
          name: entry.name,
          description: entry.description,
          requiredArgs: entry.requiredArgs as any,
          optionalArgs: entry.optionalArgs as any,
          handler: entry.handler,
        }),
        { overwrite: true }
      );
    }
  }

  /** Manually register a tool entry without waiting for refresh. */
  registerTool(entry: DynamicToolEntry): void {
    this.addChild(
      new LeafNode({
        name: entry.name,
        description: entry.description,
        requiredArgs: entry.requiredArgs as any,
        optionalArgs: entry.optionalArgs as any,
        handler: entry.handler,
      }),
      { overwrite: true }
    );
  }

  /** Remove a dynamically registered tool by name. */
  unregisterTool(name: string): boolean {
    return this.removeChild(name);
  }
}

// ── Middleware Factories ──────────────────────────────────────────────

/**
 * Creates a swiss-army-tool Middleware that emits tool execution timing
 * and results through the agent's event bus.
 *
 * This bridges swiss-army-tool's execution context into oboto-agent's
 * event-driven architecture, enabling:
 * - Real-time tool execution duration monitoring
 * - Tool execution logging / tracing
 * - Performance analytics
 *
 * ```ts
 * const timingMw = createToolTimingMiddleware(agent.bus);
 * router.use(timingMw);
 * ```
 */
export function createToolTimingMiddleware(bus: AgentEventBus): Middleware {
  return async (ctx: ExecutionContext, next: () => Promise<string>) => {
    const startTime = Date.now();
    bus.emit("tool_execution_start", {
      command: ctx.command,
      kwargs: ctx.kwargs,
      resolvedPath: ctx.resolvedPath,
    });

    try {
      const result = await next();
      const durationMs = Date.now() - startTime;

      bus.emit("tool_execution_complete", {
        command: ctx.command,
        kwargs: ctx.kwargs,
        result: result.length > 500 ? result.slice(0, 500) + "..." : result,
        durationMs,
      });

      return result;
    } catch (err) {
      const durationMs = Date.now() - startTime;
      bus.emit("error", {
        message: `Tool execution failed: ${ctx.command}`,
        error: err,
        durationMs,
      });
      throw err;
    }
  };
}

/**
 * Creates a swiss-army-tool Middleware that enforces a maximum execution
 * time for any tool call. Useful as a safety net.
 */
export function createToolTimeoutMiddleware(maxMs: number): Middleware {
  return async (_ctx: ExecutionContext, next: () => Promise<string>) => {
    const result = await Promise.race([
      next(),
      new Promise<never>((_, reject) =>
        setTimeout(
          () => reject(new Error(`Tool execution timed out after ${maxMs}ms`)),
          maxMs
        )
      ),
    ]);
    return result;
  };
}

/**
 * Creates a swiss-army-tool Middleware that logs all tool calls to the
 * session's KV store for later review. The agent can access these via
 * the memory module.
 */
export function createToolAuditMiddleware(session: SessionManager): Middleware {
  let callIndex = 0;

  return async (ctx: ExecutionContext, next: () => Promise<string>) => {
    const idx = ++callIndex;
    const startTime = Date.now();
    const result = await next();
    const durationMs = Date.now() - startTime;

    const entry = JSON.stringify({
      command: ctx.command,
      kwargs: ctx.kwargs,
      resultLength: result.length,
      durationMs,
      timestamp: new Date().toISOString(),
    });

    session.kvStore.set(`_audit:${idx}`, entry);
    return result;
  };
}

// ── Tool Tree Builder ────────────────────────────────────────────────

/**
 * Build a pre-wired swiss-army-tool command tree that includes:
 * - Memory module (set/get/list/search/pin/unpin/delete/tag)
 * - Dynamic tool branches from providers
 * - Any additional static branches
 *
 * Returns the root BranchNode, ready to be passed to `new Router(root, session)`.
 *
 * ```ts
 * const root = createAgentToolTree({
 *   session,
 *   includeMemory: true,
 *   dynamicProviders: [
 *     { name: "mcp", description: "MCP tools", provider: mcpProvider },
 *   ],
 *   staticBranches: [myFilesystemBranch, myDatabaseBranch],
 * });
 * const router = new Router(root, session);
 * ```
 */
export function createAgentToolTree(config: AgentToolTreeConfig): BranchNode {
  const builder = TreeBuilder.create("root", "Agent command tree with integrated tools");

  // Add memory module
  if (config.includeMemory !== false) {
    const memoryBranch = createMemoryModule(config.session);
    builder.addBranch(memoryBranch);
  }

  // Build the root first so we can add dynamic branches
  const root = builder.build();

  // Add dynamic tool providers
  if (config.dynamicProviders) {
    for (const dp of config.dynamicProviders) {
      const dynamicBranch = new AgentDynamicTools({
        name: dp.name,
        description: dp.description,
        provider: dp.provider,
        ttlMs: dp.ttlMs,
      });
      root.addChild(dynamicBranch);
    }
  }

  // Add additional static branches
  if (config.staticBranches) {
    for (const branch of config.staticBranches) {
      root.addChild(branch);
    }
  }

  return root;
}
