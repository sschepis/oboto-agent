import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  PermissionGuard,
  SessionCompactor,
  HookIntegration,
  SlashCommandRegistry,
  AgentUsageTracker,
} from "../adapters/as-agent-features.js";
import { AgentEventBus } from "../event-bus.js";
import { MessageRole, PermissionMode } from "@sschepis/as-agent";
import type {
  PermissionPolicy,
  PermissionPrompter,
  HookRunner,
  Session,
  CompactionConfig,
  TokenUsage,
} from "@sschepis/as-agent";

// ── Mock factories ────────────────────────────────────────────────────

function createMockPolicy(overrides?: Partial<PermissionPolicy>): PermissionPolicy {
  return {
    activeMode: PermissionMode.Allow,
    requiredModeFor: (_toolName: string) => PermissionMode.ReadOnly,
    authorize: (_toolName: string, _input: string) => ({ kind: "allow" }),
    ...overrides,
  };
}

function createMockPrompter(): PermissionPrompter {
  return {
    decide: () => ({ kind: "allow" }),
  };
}

function createMockHookRunner(overrides?: {
  preResult?: { denied: boolean; messages: string[] };
  postResult?: { denied: boolean; messages: string[] };
}): HookRunner {
  return {
    runPreToolUse: () => overrides?.preResult ?? { denied: false, messages: [] },
    runPostToolUse: () => overrides?.postResult ?? { denied: false, messages: [] },
  };
}

function createTestSession(messageCount: number): Session {
  const messages = Array.from({ length: messageCount }, (_, i) => ({
    role: i % 2 === 0 ? MessageRole.User : MessageRole.Assistant,
    blocks: [{ kind: "text" as const, text: `Message ${i + 1}: ${"x".repeat(100)}` }],
  }));
  return { version: 1, messages };
}

// ── PermissionGuard ──────────────────────────────────────────────────

describe("PermissionGuard", () => {
  it("allows tool calls when policy permits", () => {
    const policy = createMockPolicy();
    const guard = new PermissionGuard(policy, null);

    const outcome = guard.checkPermission("echo", "{}");
    expect(outcome.kind).toBe("allow");
  });

  it("denies tool calls when policy denies", () => {
    const policy = createMockPolicy({
      authorize: () => ({ kind: "deny", reason: "not allowed" }),
    });
    const bus = new AgentEventBus();
    const events: any[] = [];
    bus.on("permission_denied", (e) => events.push(e.payload));

    const guard = new PermissionGuard(policy, null, bus);
    const outcome = guard.checkPermission("bash", '{"cmd":"rm -rf /"}');

    expect(outcome.kind).toBe("deny");
    expect(outcome.reason).toBe("not allowed");
    expect(events).toHaveLength(1);
    expect(events[0].toolName).toBe("bash");
  });

  it("exposes activeMode from the policy", () => {
    const policy = createMockPolicy({ activeMode: PermissionMode.ReadOnly });
    const guard = new PermissionGuard(policy, null);
    expect(guard.activeMode).toBe(PermissionMode.ReadOnly);
  });

  it("exposes requiredModeFor from the policy", () => {
    const policy = createMockPolicy({
      requiredModeFor: () => PermissionMode.DangerFullAccess,
    });
    const guard = new PermissionGuard(policy, null);
    expect(guard.requiredModeFor("bash")).toBe(PermissionMode.DangerFullAccess);
  });

  it("passes prompter to policy.authorize", () => {
    const authorizeSpy = vi.fn(() => ({ kind: "allow" as const }));
    const policy = createMockPolicy({ authorize: authorizeSpy });
    const prompter = createMockPrompter();

    const guard = new PermissionGuard(policy, prompter);
    guard.checkPermission("test", "{}");

    expect(authorizeSpy).toHaveBeenCalledWith("test", "{}", prompter);
  });
});

// ── SessionCompactor ──────────────────────────────────────────────────

describe("SessionCompactor", () => {
  const defaultConfig: CompactionConfig = {
    preserveRecentMessages: 5,
    maxEstimatedTokens: 1000,
  };

  it("returns null when session is within token limit", () => {
    const compactor = new SessionCompactor(undefined, defaultConfig);
    const session = createTestSession(3);

    const result = compactor.compactIfNeeded(session, 500);
    expect(result).toBeNull();
  });

  it("compacts when token estimate exceeds limit", () => {
    const bus = new AgentEventBus();
    const events: any[] = [];
    bus.on("session_compacted", (e) => events.push(e.payload));

    const compactor = new SessionCompactor(bus, {
      preserveRecentMessages: 3,
      maxEstimatedTokens: 100,
    });
    const session = createTestSession(10);

    const result = compactor.compactIfNeeded(session, 500);
    expect(result).not.toBeNull();
    expect(result!.removedMessageCount).toBe(7);
    expect(result!.compactedSession.messages).toHaveLength(4); // 1 summary + 3 preserved
    expect(events).toHaveLength(1);
    expect(events[0].removedMessageCount).toBe(7);
  });

  it("does not compact when there are fewer messages than preserve count", () => {
    const compactor = new SessionCompactor(undefined, {
      preserveRecentMessages: 10,
      maxEstimatedTokens: 100,
    });
    const session = createTestSession(5);

    const result = compactor.compact(session);
    expect(result.removedMessageCount).toBe(0);
    expect(result.compactedSession).toBe(session);
  });

  it("tracks compaction stats", () => {
    const compactor = new SessionCompactor(undefined, {
      preserveRecentMessages: 2,
      maxEstimatedTokens: 50,
    });

    compactor.compact(createTestSession(10));
    compactor.compact(createTestSession(8));

    expect(compactor.stats.compactionCount).toBe(2);
    expect(compactor.stats.totalRemovedMessages).toBe(8 + 6);
  });

  it("updateConfig changes settings", () => {
    const compactor = new SessionCompactor(undefined, defaultConfig);
    compactor.updateConfig({ preserveRecentMessages: 10 });

    const session = createTestSession(12);
    const result = compactor.compact(session);
    expect(result.removedMessageCount).toBe(2); // 12 - 10 = 2
  });

  it("compacted session starts with a summary message", () => {
    const compactor = new SessionCompactor(undefined, {
      preserveRecentMessages: 2,
      maxEstimatedTokens: 50,
    });
    const session = createTestSession(6);

    const result = compactor.compact(session);
    const firstMsg = result.compactedSession.messages[0];
    expect(firstMsg.role).toBe(MessageRole.System);
    expect(firstMsg.blocks[0].kind).toBe("text");
    expect((firstMsg.blocks[0] as any).text).toContain("compacted");
  });
});

// ── HookIntegration ──────────────────────────────────────────────────

describe("HookIntegration", () => {
  it("runs pre-tool-use hooks and returns result", () => {
    const runner = createMockHookRunner();
    const hooks = new HookIntegration(runner);

    const result = hooks.runPreToolUse("echo", "{}");
    expect(result.denied).toBe(false);
    expect(result.messages).toEqual([]);
  });

  it("emits hook_denied event when pre-hook denies", () => {
    const runner = createMockHookRunner({
      preResult: { denied: true, messages: ["blocked by policy"] },
    });
    const bus = new AgentEventBus();
    const events: any[] = [];
    bus.on("hook_denied", (e) => events.push(e.payload));

    const hooks = new HookIntegration(runner, bus);
    const result = hooks.runPreToolUse("bash", '{"cmd":"rm"}');

    expect(result.denied).toBe(true);
    expect(events).toHaveLength(1);
    expect(events[0].phase).toBe("pre");
    expect(events[0].toolName).toBe("bash");
  });

  it("runs post-tool-use hooks", () => {
    const runner = createMockHookRunner({
      postResult: { denied: false, messages: ["logged"] },
    });
    const bus = new AgentEventBus();
    const events: any[] = [];
    bus.on("hook_message", (e) => events.push(e.payload));

    const hooks = new HookIntegration(runner, bus);
    hooks.runPostToolUse("echo", "{}", "result", false);

    expect(events).toHaveLength(1);
    expect(events[0].phase).toBe("post");
    expect(events[0].messages).toEqual(["logged"]);
  });
});

// ── SlashCommandRegistry ─────────────────────────────────────────────

describe("SlashCommandRegistry", () => {
  it("constructs without Wasm runtime", () => {
    const registry = new SlashCommandRegistry();
    expect(registry.getCommandSpecs()).toEqual([]);
  });

  it("registers and executes custom commands", async () => {
    const registry = new SlashCommandRegistry();
    registry.registerCommand(
      { name: "status", summary: "Show status", argumentHint: "", resumeSupported: false },
      (args) => `Status: ${args || "ok"}`,
    );

    const result = await registry.executeCustomCommand("status", "");
    expect(result).toBe("Status: ok");
  });

  it("returns null for unknown custom commands", async () => {
    const registry = new SlashCommandRegistry();
    const result = await registry.executeCustomCommand("unknown", "");
    expect(result).toBeNull();
  });

  it("unregisters custom commands", () => {
    const registry = new SlashCommandRegistry();
    registry.registerCommand(
      { name: "test", summary: "Test", argumentHint: "", resumeSupported: false },
      () => "test",
    );

    expect(registry.unregisterCommand("test")).toBe(true);
    expect(registry.unregisterCommand("test")).toBe(false);
  });

  it("parses slash commands from input", () => {
    const registry = new SlashCommandRegistry();

    expect(registry.parseCommand("/help")).toEqual({ name: "help", args: "" });
    expect(registry.parseCommand("/model gpt-4")).toEqual({ name: "model", args: "gpt-4" });
    expect(registry.parseCommand("hello")).toBeNull();
    expect(registry.parseCommand("")).toBeNull();
  });

  it("includes custom commands in help text", () => {
    const registry = new SlashCommandRegistry();
    registry.registerCommand(
      { name: "status", summary: "Show agent status", argumentHint: "", resumeSupported: false },
      () => "ok",
    );

    const help = registry.getHelpText();
    expect(help).toContain("status");
    expect(help).toContain("Show agent status");
  });

  it("hasCommand checks custom commands", () => {
    const registry = new SlashCommandRegistry();
    registry.registerCommand(
      { name: "test", summary: "Test", argumentHint: "", resumeSupported: false },
      () => "ok",
    );

    expect(registry.hasCommand("test")).toBe(true);
    expect(registry.hasCommand("nonexistent")).toBe(false);
  });

  it("getResumeSupportedCommands filters correctly", () => {
    const registry = new SlashCommandRegistry();
    registry.registerCommand(
      { name: "resumable", summary: "Resumable", argumentHint: "", resumeSupported: true },
      () => "ok",
    );
    registry.registerCommand(
      { name: "nonresumable", summary: "Not resumable", argumentHint: "", resumeSupported: false },
      () => "ok",
    );

    const resumable = registry.getResumeSupportedCommands();
    expect(resumable).toHaveLength(1);
    expect(resumable[0].name).toBe("resumable");
  });
});

// ── AgentUsageTracker ────────────────────────────────────────────────

describe("AgentUsageTracker", () => {
  it("starts with zero usage", () => {
    const tracker = new AgentUsageTracker();
    const usage = tracker.cumulativeUsage();
    expect(usage.inputTokens).toBe(0);
    expect(usage.outputTokens).toBe(0);
    expect(tracker.turns()).toBe(0);
  });

  it("records usage for current turn", () => {
    const tracker = new AgentUsageTracker();
    tracker.record({ inputTokens: 100, outputTokens: 50, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });

    const current = tracker.currentTurnUsage();
    expect(current.inputTokens).toBe(100);
    expect(current.outputTokens).toBe(50);
  });

  it("accumulates usage within a turn", () => {
    const tracker = new AgentUsageTracker();
    tracker.record({ inputTokens: 100, outputTokens: 50, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });
    tracker.record({ inputTokens: 200, outputTokens: 100, cacheCreationInputTokens: 10, cacheReadInputTokens: 5 });

    const current = tracker.currentTurnUsage();
    expect(current.inputTokens).toBe(300);
    expect(current.outputTokens).toBe(150);
    expect(current.cacheCreationInputTokens).toBe(10);
    expect(current.cacheReadInputTokens).toBe(5);
  });

  it("endTurn finalizes the turn and starts a new one", () => {
    const tracker = new AgentUsageTracker();
    tracker.record({ inputTokens: 100, outputTokens: 50, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });
    tracker.endTurn();

    const current = tracker.currentTurnUsage();
    expect(current.inputTokens).toBe(0);
    expect(current.outputTokens).toBe(0);
    expect(tracker.turns()).toBe(1);
  });

  it("tracks cumulative usage across turns", () => {
    const tracker = new AgentUsageTracker();

    tracker.record({ inputTokens: 100, outputTokens: 50, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });
    tracker.endTurn();

    tracker.record({ inputTokens: 200, outputTokens: 100, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });
    tracker.endTurn();

    const cumulative = tracker.cumulativeUsage();
    expect(cumulative.inputTokens).toBe(300);
    expect(cumulative.outputTokens).toBe(150);
    expect(tracker.turns()).toBe(2);
  });

  it("includes current turn in cumulative", () => {
    const tracker = new AgentUsageTracker();
    tracker.record({ inputTokens: 100, outputTokens: 50, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });
    tracker.endTurn();
    tracker.record({ inputTokens: 50, outputTokens: 25, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });

    const cumulative = tracker.cumulativeUsage();
    expect(cumulative.inputTokens).toBe(150);
    expect(cumulative.outputTokens).toBe(75);
    expect(tracker.turns()).toBe(2); // 1 completed + 1 active
  });

  it("reset clears all data", () => {
    const tracker = new AgentUsageTracker();
    tracker.record({ inputTokens: 100, outputTokens: 50, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });
    tracker.endTurn();
    tracker.reset();

    expect(tracker.turns()).toBe(0);
    expect(tracker.cumulativeUsage().inputTokens).toBe(0);
    expect(tracker.currentTurnUsage().inputTokens).toBe(0);
  });

  it("endTurn on empty turn does not increment turn count", () => {
    const tracker = new AgentUsageTracker();
    tracker.endTurn();
    tracker.endTurn();
    expect(tracker.turns()).toBe(0);
  });
});
