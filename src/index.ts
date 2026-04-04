// ── oboto-agent: Event-driven dual-LLM orchestration ───────────────

// Main orchestrator
export { ObotoAgent } from "./oboto-agent.js";

// Types
export type {
  ObotoAgentConfig,
  AgentEventType,
  AgentEvent,
  TriageResult,
  ToolExecutionEvent,
} from "./types.js";

// Event bus
export { AgentEventBus } from "./event-bus.js";

// Adapters
export { createRouterTool } from "./adapters/tools.js";
export { toChat, fromChat, sessionToHistory, createEmptySession } from "./adapters/memory.js";
export { toLmscriptProvider } from "./adapters/llm-wrapper.js";

// Context management
export { ContextManager } from "./context-manager.js";

// Triage
export { createTriageFunction, TriageSchema } from "./triage.js";
