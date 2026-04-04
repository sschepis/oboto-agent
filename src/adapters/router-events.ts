/**
 * router-events.ts — Bridge LLMRouter health/failover events into the agent event bus.
 *
 * The LLMRouter (from @sschepis/llm-wrapper) exposes a `RouterEventEmitter` that
 * fires typed events for routing decisions, fallbacks, circuit breaker state changes,
 * and request completions/errors.
 *
 * This module:
 * 1. Subscribes to all LLMRouter events
 * 2. Forwards them to the agent's AgentEventBus as `router_event` events
 * 3. Provides a health snapshot method for observability
 */

import type {
  LLMRouter,
  RouterEventMap,
  HealthState,
} from "@sschepis/llm-wrapper";
import type { AgentEventBus } from "../event-bus.js";

/** All LLMRouter event names we subscribe to. */
const ROUTER_EVENTS: (keyof RouterEventMap)[] = [
  "route",
  "fallback",
  "circuit:open",
  "circuit:close",
  "circuit:half-open",
  "request:complete",
  "request:error",
];

/**
 * Bridge that forwards LLMRouter events to the agent event bus.
 *
 * Usage:
 * ```ts
 * const bridge = new RouterEventBridge(bus);
 * bridge.attach(localRouter, "local");
 * bridge.attach(remoteRouter, "remote");
 *
 * // Later, get a health snapshot
 * const health = bridge.getHealthSnapshot();
 * ```
 */
export class RouterEventBridge {
  private bus: AgentEventBus;
  private attachedRouters = new Map<string, {
    router: LLMRouter;
    detachFns: Array<() => void>;
  }>();

  constructor(bus: AgentEventBus) {
    this.bus = bus;
  }

  /**
   * Attach an LLMRouter and forward all its events to the agent bus.
   *
   * @param router - The LLMRouter to monitor
   * @param label - A label to identify this router in events (e.g. "local", "remote")
   */
  attach(router: LLMRouter, label: string): void {
    // Detach if already attached with this label
    this.detach(label);

    const detachFns: Array<() => void> = [];

    for (const eventName of ROUTER_EVENTS) {
      const handler = (data: RouterEventMap[typeof eventName]) => {
        this.bus.emit("router_event", {
          routerLabel: label,
          eventName,
          data,
          timestamp: Date.now(),
        });
      };

      router.events.on(eventName, handler as any);
      detachFns.push(() => {
        router.events.off(eventName, handler as any);
      });
    }

    this.attachedRouters.set(label, { router, detachFns });
  }

  /**
   * Detach a previously attached router.
   */
  detach(label: string): void {
    const entry = this.attachedRouters.get(label);
    if (entry) {
      for (const fn of entry.detachFns) {
        fn();
      }
      this.attachedRouters.delete(label);
    }
  }

  /**
   * Detach all attached routers.
   */
  detachAll(): void {
    for (const label of this.attachedRouters.keys()) {
      this.detach(label);
    }
  }

  /**
   * Get a health snapshot from all attached routers.
   * Returns a map of router label -> endpoint name -> HealthState.
   */
  getHealthSnapshot(): Map<string, Map<string, HealthState>> {
    const snapshot = new Map<string, Map<string, HealthState>>();

    for (const [label, { router }] of this.attachedRouters) {
      snapshot.set(label, router.getHealthState());
    }

    return snapshot;
  }

  /**
   * Check if any attached router has an endpoint in "open" (tripped) circuit state.
   */
  hasTrippedCircuits(): boolean {
    for (const [, { router }] of this.attachedRouters) {
      for (const [, health] of router.getHealthState()) {
        if (health.status === "open") return true;
      }
    }
    return false;
  }

  /**
   * Get the labels of all attached routers.
   */
  get labels(): string[] {
    return Array.from(this.attachedRouters.keys());
  }
}

/**
 * Helper to check if a ProviderLike is an LLMRouter (has `events` property).
 */
export function isLLMRouter(provider: unknown): provider is LLMRouter {
  return (
    typeof provider === "object" &&
    provider !== null &&
    "events" in provider &&
    "getHealthState" in provider
  );
}
