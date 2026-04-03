import { describe, it, expect, vi } from "vitest";
import { AgentEventBus } from "../event-bus.js";
import type { AgentEvent } from "../types.js";

describe("AgentEventBus", () => {
  it("emits events to subscribers", () => {
    const bus = new AgentEventBus();
    const handler = vi.fn();

    bus.on("user_input", handler);
    bus.emit("user_input", { text: "hello" });

    expect(handler).toHaveBeenCalledOnce();
    const event: AgentEvent = handler.mock.calls[0][0];
    expect(event.type).toBe("user_input");
    expect(event.payload).toEqual({ text: "hello" });
    expect(event.timestamp).toBeTypeOf("number");
  });

  it("supports multiple handlers for the same event", () => {
    const bus = new AgentEventBus();
    const handler1 = vi.fn();
    const handler2 = vi.fn();

    bus.on("agent_thought", handler1);
    bus.on("agent_thought", handler2);
    bus.emit("agent_thought", { text: "thinking" });

    expect(handler1).toHaveBeenCalledOnce();
    expect(handler2).toHaveBeenCalledOnce();
  });

  it("does not cross-emit between event types", () => {
    const bus = new AgentEventBus();
    const handler = vi.fn();

    bus.on("user_input", handler);
    bus.emit("error", { message: "oops" });

    expect(handler).not.toHaveBeenCalled();
  });

  it("returns an unsubscribe function from on()", () => {
    const bus = new AgentEventBus();
    const handler = vi.fn();

    const unsub = bus.on("user_input", handler);
    unsub();
    bus.emit("user_input", { text: "hello" });

    expect(handler).not.toHaveBeenCalled();
  });

  it("supports off() to remove a handler", () => {
    const bus = new AgentEventBus();
    const handler = vi.fn();

    bus.on("error", handler);
    bus.off("error", handler);
    bus.emit("error", { message: "test" });

    expect(handler).not.toHaveBeenCalled();
  });

  it("once() fires handler only once", () => {
    const bus = new AgentEventBus();
    const handler = vi.fn();

    bus.once("turn_complete", handler);
    bus.emit("turn_complete", { model: "local" });
    bus.emit("turn_complete", { model: "remote" });

    expect(handler).toHaveBeenCalledOnce();
    expect(handler.mock.calls[0][0].payload).toEqual({ model: "local" });
  });

  it("removeAllListeners() clears everything", () => {
    const bus = new AgentEventBus();
    const h1 = vi.fn();
    const h2 = vi.fn();

    bus.on("user_input", h1);
    bus.on("error", h2);
    bus.removeAllListeners();
    bus.emit("user_input", {});
    bus.emit("error", {});

    expect(h1).not.toHaveBeenCalled();
    expect(h2).not.toHaveBeenCalled();
  });

  it("handles emit with no subscribers gracefully", () => {
    const bus = new AgentEventBus();
    expect(() => bus.emit("interruption", {})).not.toThrow();
  });
});
