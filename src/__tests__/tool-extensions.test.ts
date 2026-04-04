import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  AgentDynamicTools,
  createToolTimingMiddleware,
  createToolTimeoutMiddleware,
  createToolAuditMiddleware,
  createAgentToolTree,
  type DynamicToolProvider,
  type DynamicToolEntry,
} from "../adapters/tool-extensions.js";
import {
  BranchNode,
  LeafNode,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";
import { AgentEventBus } from "../event-bus.js";

// ── AgentDynamicTools ─────────────────────────────────────────────────

describe("AgentDynamicTools", () => {
  it("constructs with a provider", () => {
    const provider: DynamicToolProvider = {
      discover: () => [],
    };
    const dynamic = new AgentDynamicTools({
      name: "test",
      description: "Test dynamic tools",
      provider,
    });
    expect(dynamic).toBeDefined();
  });

  it("registerTool adds a child leaf node", () => {
    const provider: DynamicToolProvider = { discover: () => [] };
    const dynamic = new AgentDynamicTools({
      name: "test",
      description: "Test",
      provider,
    });

    dynamic.registerTool({
      name: "mytool",
      description: "A test tool",
      handler: () => "result",
    });

    expect(dynamic.children.has("mytool")).toBe(true);
  });

  it("unregisterTool removes a child", () => {
    const provider: DynamicToolProvider = { discover: () => [] };
    const dynamic = new AgentDynamicTools({
      name: "test",
      description: "Test",
      provider,
    });

    dynamic.registerTool({
      name: "mytool",
      description: "A test tool",
      handler: () => "result",
    });

    const removed = dynamic.unregisterTool("mytool");
    expect(removed).toBe(true);
    expect(dynamic.children.has("mytool")).toBe(false);
  });

  it("unregisterTool returns false for non-existent tool", () => {
    const provider: DynamicToolProvider = { discover: () => [] };
    const dynamic = new AgentDynamicTools({
      name: "test",
      description: "Test",
      provider,
    });

    const removed = dynamic.unregisterTool("nonexistent");
    expect(removed).toBe(false);
  });

  it("registerTool overwrites existing tool", () => {
    const provider: DynamicToolProvider = { discover: () => [] };
    const dynamic = new AgentDynamicTools({
      name: "test",
      description: "Test",
      provider,
    });

    dynamic.registerTool({
      name: "mytool",
      description: "Version 1",
      handler: () => "v1",
    });

    dynamic.registerTool({
      name: "mytool",
      description: "Version 2",
      handler: () => "v2",
    });

    const tool = dynamic.children.get("mytool");
    expect(tool?.description).toBe("Version 2");
  });
});

// ── Middleware Factories ──────────────────────────────────────────────

describe("createToolTimingMiddleware", () => {
  it("emits tool_execution_start and tool_execution_complete events", async () => {
    const bus = new AgentEventBus();
    const events: { type: string; payload: any }[] = [];
    bus.on("tool_execution_start", (e) => events.push({ type: "start", payload: e.payload }));
    bus.on("tool_execution_complete", (e) => events.push({ type: "complete", payload: e.payload }));

    const mw = createToolTimingMiddleware(bus);
    const ctx = {
      command: "echo",
      kwargs: { text: "hello" },
      resolvedPath: "root/echo",
      session: new SessionManager("test"),
    } as any;

    const result = await mw(ctx, async () => "echo result");
    expect(result).toBe("echo result");

    expect(events).toHaveLength(2);
    expect(events[0].type).toBe("start");
    expect((events[0].payload as any).command).toBe("echo");
    expect(events[1].type).toBe("complete");
    expect((events[1].payload as any).durationMs).toBeTypeOf("number");
  });

  it("emits error event on tool failure", async () => {
    const bus = new AgentEventBus();
    const errors: any[] = [];
    bus.on("error", (e) => errors.push(e.payload));

    const mw = createToolTimingMiddleware(bus);
    const ctx = {
      command: "fail",
      kwargs: {},
      resolvedPath: "root/fail",
      session: new SessionManager("test"),
    } as any;

    await expect(
      mw(ctx, async () => { throw new Error("tool broke"); }),
    ).rejects.toThrow("tool broke");

    expect(errors).toHaveLength(1);
    expect((errors[0] as any).message).toContain("fail");
  });
});

describe("createToolTimeoutMiddleware", () => {
  it("allows fast tool calls through", async () => {
    const mw = createToolTimeoutMiddleware(5000);
    const ctx = {} as any;

    const result = await mw(ctx, async () => "fast");
    expect(result).toBe("fast");
  });

  it("rejects slow tool calls", async () => {
    const mw = createToolTimeoutMiddleware(50);
    const ctx = {} as any;

    await expect(
      mw(ctx, () => new Promise((resolve) => setTimeout(() => resolve("slow"), 200))),
    ).rejects.toThrow("timed out");
  });
});

describe("createToolAuditMiddleware", () => {
  it("logs tool calls to session KV store", async () => {
    const session = new SessionManager("test");
    const mw = createToolAuditMiddleware(session);
    const ctx = {
      command: "echo",
      kwargs: { text: "hello" },
    } as any;

    await mw(ctx, async () => "result");

    // The audit middleware uses incrementing keys: _audit:1, _audit:2, etc.
    const entry = session.kvStore.get("_audit:1");
    expect(entry).toBeDefined();

    const parsed = JSON.parse(entry!);
    expect(parsed.command).toBe("echo");
    expect(parsed.resultLength).toBe(6); // "result".length
    expect(parsed.durationMs).toBeTypeOf("number");
  });
});

// ── createAgentToolTree ──────────────────────────────────────────────

describe("createAgentToolTree", () => {
  it("creates a root with memory module by default", () => {
    const session = new SessionManager("test");
    const root = createAgentToolTree({ session });
    expect(root.children.has("memory")).toBe(true);
  });

  it("skips memory module when includeMemory is false", () => {
    const session = new SessionManager("test");
    const root = createAgentToolTree({ session, includeMemory: false });
    expect(root.children.has("memory")).toBe(false);
  });

  it("adds static branches", () => {
    const session = new SessionManager("test");
    const customBranch = new BranchNode({ name: "custom", description: "Custom branch" });
    customBranch.addChild(new LeafNode({
      name: "test",
      description: "Test leaf",
      handler: () => "test",
    }));

    const root = createAgentToolTree({
      session,
      staticBranches: [customBranch],
    });

    expect(root.children.has("custom")).toBe(true);
  });

  it("adds dynamic providers as branches", () => {
    const session = new SessionManager("test");
    const provider: DynamicToolProvider = { discover: () => [] };

    const root = createAgentToolTree({
      session,
      dynamicProviders: [
        { name: "dynamic", description: "Dynamic tools", provider },
      ],
    });

    expect(root.children.has("dynamic")).toBe(true);
  });
});
