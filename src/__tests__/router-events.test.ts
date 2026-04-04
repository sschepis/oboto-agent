import { describe, it, expect, vi, beforeEach } from "vitest";
import { RouterEventBridge, isLLMRouter } from "../adapters/router-events.js";
import { AgentEventBus } from "../event-bus.js";
import type { AgentEvent } from "../types.js";

// ── Mock LLMRouter ───────────────────────────────────────────────────

class MockRouterEventEmitter {
  private handlers = new Map<string, Set<Function>>();

  on(event: string, handler: Function) {
    if (!this.handlers.has(event)) this.handlers.set(event, new Set());
    this.handlers.get(event)!.add(handler);
    return this;
  }

  off(event: string, handler: Function) {
    this.handlers.get(event)?.delete(handler);
    return this;
  }

  emit(event: string, data: any) {
    const handlers = this.handlers.get(event);
    if (handlers) {
      for (const handler of handlers) handler(data);
    }
  }

  removeAllListeners() {
    this.handlers.clear();
    return this;
  }

  getHandlerCount(event: string): number {
    return this.handlers.get(event)?.size ?? 0;
  }
}

function createMockRouter(healthState?: Map<string, any>) {
  const emitter = new MockRouterEventEmitter();
  return {
    events: emitter,
    getHealthState: () => healthState ?? new Map(),
    chat: vi.fn(),
    stream: vi.fn(),
    addEndpoint: vi.fn(),
    removeEndpoint: vi.fn(),
    resetHealth: vi.fn(),
  };
}

// ── RouterEventBridge ────────────────────────────────────────────────

describe("RouterEventBridge", () => {
  let bus: AgentEventBus;
  let bridge: RouterEventBridge;

  beforeEach(() => {
    bus = new AgentEventBus();
    bridge = new RouterEventBridge(bus);
  });

  it("attaches to a router and forwards events", () => {
    const router = createMockRouter();
    const events: AgentEvent[] = [];
    bus.on("router_event", (e) => events.push(e));

    bridge.attach(router as any, "remote");

    // Simulate a route event from the router
    router.events.emit("route", {
      decision: { endpoint: { name: "gpt-4" }, reason: "priority" },
      context: {},
    });

    expect(events).toHaveLength(1);
    expect((events[0].payload as any).routerLabel).toBe("remote");
    expect((events[0].payload as any).eventName).toBe("route");
    expect((events[0].payload as any).data.decision.endpoint.name).toBe("gpt-4");
  });

  it("attaches to multiple routers with different labels", () => {
    const localRouter = createMockRouter();
    const remoteRouter = createMockRouter();
    const events: AgentEvent[] = [];
    bus.on("router_event", (e) => events.push(e));

    bridge.attach(localRouter as any, "local");
    bridge.attach(remoteRouter as any, "remote");

    localRouter.events.emit("request:complete", { endpoint: { name: "llama" }, latencyMs: 50 });
    remoteRouter.events.emit("request:complete", { endpoint: { name: "gpt-4" }, latencyMs: 200 });

    expect(events).toHaveLength(2);
    expect((events[0].payload as any).routerLabel).toBe("local");
    expect((events[1].payload as any).routerLabel).toBe("remote");
  });

  it("detaches a router and stops forwarding events", () => {
    const router = createMockRouter();
    const events: AgentEvent[] = [];
    bus.on("router_event", (e) => events.push(e));

    bridge.attach(router as any, "remote");
    bridge.detach("remote");

    router.events.emit("route", { decision: {}, context: {} });
    expect(events).toHaveLength(0);
  });

  it("detachAll removes all subscriptions", () => {
    const router1 = createMockRouter();
    const router2 = createMockRouter();
    const events: AgentEvent[] = [];
    bus.on("router_event", (e) => events.push(e));

    bridge.attach(router1 as any, "r1");
    bridge.attach(router2 as any, "r2");
    bridge.detachAll();

    router1.events.emit("route", {});
    router2.events.emit("route", {});
    expect(events).toHaveLength(0);
    expect(bridge.labels).toEqual([]);
  });

  it("re-attaching with same label detaches the old one first", () => {
    const router1 = createMockRouter();
    const router2 = createMockRouter();
    const events: AgentEvent[] = [];
    bus.on("router_event", (e) => events.push(e));

    bridge.attach(router1 as any, "remote");
    bridge.attach(router2 as any, "remote");

    // Events from router1 should no longer come through
    router1.events.emit("route", { from: "r1" });
    expect(events).toHaveLength(0);

    // Events from router2 should come through
    router2.events.emit("route", { from: "r2" });
    expect(events).toHaveLength(1);
    expect((events[0].payload as any).data.from).toBe("r2");
  });

  it("forwards circuit breaker events", () => {
    const router = createMockRouter();
    const events: AgentEvent[] = [];
    bus.on("router_event", (e) => events.push(e));

    bridge.attach(router as any, "remote");

    router.events.emit("circuit:open", {
      endpoint: { name: "gpt-4" },
      health: { status: "open", errorRate: 0.8, consecutiveFailures: 5 },
    });

    expect(events).toHaveLength(1);
    expect((events[0].payload as any).eventName).toBe("circuit:open");
  });

  it("forwards fallback events", () => {
    const router = createMockRouter();
    const events: AgentEvent[] = [];
    bus.on("router_event", (e) => events.push(e));

    bridge.attach(router as any, "remote");

    router.events.emit("fallback", {
      from: { name: "gpt-4" },
      to: { name: "claude" },
      error: { message: "timeout" },
      attempt: 2,
    });

    expect(events).toHaveLength(1);
    expect((events[0].payload as any).eventName).toBe("fallback");
    expect((events[0].payload as any).data.attempt).toBe(2);
  });

  it("getHealthSnapshot returns health from all attached routers", () => {
    const health1 = new Map([
      ["llama", { status: "closed" as const, errorRate: 0, avgLatencyMs: 50, consecutiveFailures: 0, totalRequests: 100, totalErrors: 0 }],
    ]);
    const health2 = new Map([
      ["gpt-4", { status: "closed" as const, errorRate: 0.1, avgLatencyMs: 200, consecutiveFailures: 0, totalRequests: 50, totalErrors: 5 }],
    ]);

    const router1 = createMockRouter(health1);
    const router2 = createMockRouter(health2);

    bridge.attach(router1 as any, "local");
    bridge.attach(router2 as any, "remote");

    const snapshot = bridge.getHealthSnapshot();
    expect(snapshot.size).toBe(2);
    expect(snapshot.get("local")?.get("llama")?.status).toBe("closed");
    expect(snapshot.get("remote")?.get("gpt-4")?.errorRate).toBe(0.1);
  });

  it("hasTrippedCircuits detects open circuits", () => {
    const healthWithOpen = new Map([
      ["gpt-4", { status: "open" as const, errorRate: 0.8, avgLatencyMs: 0, consecutiveFailures: 5, totalRequests: 10, totalErrors: 8 }],
    ]);
    const router = createMockRouter(healthWithOpen);

    bridge.attach(router as any, "remote");
    expect(bridge.hasTrippedCircuits()).toBe(true);
  });

  it("hasTrippedCircuits returns false when all circuits are closed", () => {
    const healthClosed = new Map([
      ["gpt-4", { status: "closed" as const, errorRate: 0, avgLatencyMs: 100, consecutiveFailures: 0, totalRequests: 50, totalErrors: 0 }],
    ]);
    const router = createMockRouter(healthClosed);

    bridge.attach(router as any, "remote");
    expect(bridge.hasTrippedCircuits()).toBe(false);
  });

  it("labels returns attached router labels", () => {
    const router1 = createMockRouter();
    const router2 = createMockRouter();

    bridge.attach(router1 as any, "local");
    bridge.attach(router2 as any, "remote");

    expect(bridge.labels).toEqual(["local", "remote"]);
  });
});

// ── isLLMRouter ──────────────────────────────────────────────────────

describe("isLLMRouter", () => {
  it("returns true for objects with events and getHealthState", () => {
    const router = {
      events: new MockRouterEventEmitter(),
      getHealthState: () => new Map(),
    };
    expect(isLLMRouter(router)).toBe(true);
  });

  it("returns false for plain objects", () => {
    expect(isLLMRouter({})).toBe(false);
    expect(isLLMRouter({ chat: () => {} })).toBe(false);
  });

  it("returns false for null/undefined", () => {
    expect(isLLMRouter(null)).toBe(false);
    expect(isLLMRouter(undefined)).toBe(false);
  });

  it("returns false for BaseProvider-like objects without events", () => {
    expect(isLLMRouter({
      providerName: "mock",
      chat: () => {},
      stream: () => {},
    })).toBe(false);
  });
});
