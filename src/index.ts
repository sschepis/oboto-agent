// ── oboto-agent: Event-driven dual-LLM orchestration ───────────────

// Main orchestrator
export { ObotoAgent } from "./oboto-agent.js";

// Types
export type {
  ObotoAgentConfig,
  AgentEventType,
  AgentEvent,
  AgentPhase,
  TriageResult,
  ToolExecutionEvent,
  ProviderLike,
  PhaseEvent,
  DoomLoopEvent,
  ToolRoundEvent,
} from "./types.js";

// Event bus
export { AgentEventBus } from "./event-bus.js";

// Adapters
export { createRouterTool } from "./adapters/tools.js";
export { toChat, fromChat, sessionToHistory, createEmptySession } from "./adapters/memory.js";
export { toLmscriptProvider } from "./adapters/llm-wrapper.js";

// Swiss-army-tool extensions
export {
  AgentDynamicTools,
  createToolTimingMiddleware,
  createToolTimeoutMiddleware,
  createToolAuditMiddleware,
  createAgentToolTree,
} from "./adapters/tool-extensions.js";
export type {
  DynamicToolProvider,
  DynamicToolEntry,
  AgentToolTreeConfig,
} from "./adapters/tool-extensions.js";

// RAG integration
export { ConversationRAG } from "./adapters/rag-integration.js";
export type {
  ConversationRAGConfig,
  RAGRetrievalResult,
} from "./adapters/rag-integration.js";

// as-agent feature integration
export {
  PermissionGuard,
  SessionCompactor,
  HookIntegration,
  SlashCommandRegistry,
  AgentUsageTracker,
} from "./adapters/as-agent-features.js";

// Router event bridge (LLMRouter observability)
export { RouterEventBridge, isLLMRouter } from "./adapters/router-events.js";

// Pipeline workflows (lmscript Pipeline integration)
export {
  createTriageStep,
  createPlanStep,
  createExecutionStep,
  createSummaryStep,
  createTriagePlanExecutePipeline,
  createFullPipeline,
  createAnalyzeRespondPipeline,
  runAgentPipeline,
  TriagePipelineSchema,
  PlanSchema,
  ExecutionSchema,
  SummarySchema,
} from "./adapters/pipeline-workflows.js";
export type {
  TriagePipelineOutput,
  PlanOutput,
  ExecutionOutput,
  SummaryOutput,
  AgentPipelineConfig,
} from "./adapters/pipeline-workflows.js";

// Usage bridge (as-agent ↔ lmscript cost/token unification)
export {
  UsageBridge,
  asTokenUsageToLmscript,
  lmscriptToAsTokenUsage,
  estimateCostFromAsAgent,
} from "./adapters/usage-bridge.js";
export type { UnifiedCostSummary } from "./adapters/usage-bridge.js";

// Context management
export { ContextManager } from "./context-manager.js";

// Triage
export { createTriageFunction, TriageSchema } from "./triage.js";
