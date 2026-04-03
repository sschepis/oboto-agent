import type { AgentEventType, AgentEvent } from "./types.js";

type EventHandler = (event: AgentEvent) => void;

/**
 * Platform-agnostic typed event bus.
 * Uses a plain Map instead of Node.js EventEmitter for browser/Deno/Bun compatibility.
 */
export class AgentEventBus {
  private listeners = new Map<AgentEventType, Set<EventHandler>>();

  /** Subscribe to an event type. Returns an unsubscribe function. */
  on(type: AgentEventType, handler: EventHandler): () => void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(handler);
    return () => this.off(type, handler);
  }

  /** Unsubscribe a handler from an event type. */
  off(type: AgentEventType, handler: EventHandler): void {
    this.listeners.get(type)?.delete(handler);
  }

  /** Subscribe to an event type for a single emission. */
  once(type: AgentEventType, handler: EventHandler): () => void {
    const wrapper: EventHandler = (event) => {
      this.off(type, wrapper);
      handler(event);
    };
    return this.on(type, wrapper);
  }

  /** Emit an event to all subscribers. */
  emit(type: AgentEventType, payload: unknown): void {
    const event: AgentEvent = {
      type,
      payload,
      timestamp: Date.now(),
    };
    const handlers = this.listeners.get(type);
    if (handlers) {
      for (const handler of handlers) {
        handler(event);
      }
    }
  }

  /** Remove all listeners for all event types. */
  removeAllListeners(): void {
    this.listeners.clear();
  }
}
