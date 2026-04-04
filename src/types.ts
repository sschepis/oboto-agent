import type { BaseProvider, LLMRouter } from "@sschepis/llm-wrapper";
import type { Router, Middleware as ToolMiddleware } from "@sschepis/swiss-army-tool";
import type {
  Session,
  PermissionPolicy,
  PermissionPrompter,
  CompactionConfig,
  HookRunner,
  AgentRuntime,
} from "@sschepis/as-agent";
import type {
  MiddlewareHooks,
  ModelPricing,
  BudgetConfig,
  RateLimitConfig,
  EmbeddingProvider,
  VectorStore,
  VectorSearchResult,
} from "@sschepis/lmscript";

// ── Provider type ──────────────────────────────────────────────────

/**
 * A provider-like object that has `chat()` and `stream()` methods.
 * Accepts both llm-wrapper `BaseProvider` and `LLMRouter`.
 */
export type ProviderLike = BaseProvider | LLMRouter;

// ── Configuration ──────────────────────────────────────────────────

export interface ObotoAgentConfig {
  /** Small, fast model for triage and summarization (e.g. Ollama, LMStudio) */
  localModel: ProviderLike;
  /** Powerful model for complex reasoning (e.g. Anthropic, OpenAI, Gemini) */
  remoteModel: ProviderLike;
  /** Model identifier for the local provider (e.g. "llama3:8b") */
  localModelName: string;
  /** Model identifier for the remote provider (e.g. "claude-sonnet-4-20250514") */
  remoteModelName: string;
  /** Pre-built swiss-army-tool Router for tool execution */
  router: Router;
  /** Existing session to resume (creates empty session if omitted) */
  session?: Session;
  /** Maximum tokens for context window management. Default: 8192 */
  maxContextTokens?: number;
  /** Maximum LLM iterations per turn. Default: 10 */
  maxIterations?: number;
  /** System prompt prepended to all LLM calls */
  systemPrompt?: string;
  /** Called with each token chunk during LLM streaming. Enables real-time output. */
  onToken?: (token: string) => void;

  // ── lmscript infrastructure options ──────────────────────────────

  /** Middleware hooks for the lmscript execution lifecycle */
  middleware?: MiddlewareHooks[];
  /** Model pricing map for cost tracking (model name -> per-1k-token costs) */
  modelPricing?: ModelPricing;
  /** Budget limits for the remote runtime */
  budget?: BudgetConfig;
  /** Rate limiting configuration for remote API calls */
  rateLimit?: RateLimitConfig;
  /** TTL in ms for execution cache. 0 or undefined disables caching. Default: 0 */
  cacheTtlMs?: number;
  /** TTL in ms for triage cache. Separate from main cache. Default: 60000 */
  triageCacheTtlMs?: number;

  // ── swiss-army-tool options ──────────────────────────────────────

  /** Middleware to apply to the swiss-army-tool Router */
  toolMiddleware?: ToolMiddleware[];

  // ── RAG options ─────────────────────────────────────────────────

  /** Embedding provider for RAG-augmented conversation retrieval */
  embeddingProvider?: EmbeddingProvider;
  /** Custom vector store for RAG. Defaults to MemoryVectorStore. */
  vectorStore?: VectorStore;
  /** Number of past context chunks to retrieve via RAG. Default: 5 */
  ragTopK?: number;
  /** Minimum similarity score for RAG retrieval (0-1). Default: 0.3 */
  ragMinScore?: number;
  /** Embedding model identifier for RAG. Default: provider's default */
  ragEmbeddingModel?: string;
  /** Whether to automatically index conversation messages for RAG. Default: true */
  ragAutoIndex?: boolean;
  /** Whether to index tool execution results for RAG. Default: true */
  ragIndexToolResults?: boolean;
  /** Custom formatter for RAG-retrieved context */
  ragFormatContext?: (results: VectorSearchResult[]) => string;

  // ── as-agent integration options ────────────────────────────────

  /** Permission policy for tool authorization (from @sschepis/as-agent) */
  permissionPolicy?: PermissionPolicy;
  /** Permission prompter for interactive permission decisions */
  permissionPrompter?: PermissionPrompter;
  /** Session compaction config — auto-compact when estimated tokens exceed limit */
  compactionConfig?: CompactionConfig;
  /** Hook runner for pre/post tool-use shell hooks */
  hookRunner?: HookRunner;
  /** as-agent Wasm runtime for slash commands and utilities */
  agentRuntime?: AgentRuntime;
}

// ── Event Bus ──────────────────────────────────────────────────────

export type AgentEventType =
  | "user_input"
  | "agent_thought"
  | "token"
  | "triage_result"
  | "tool_execution_start"
  | "tool_execution_complete"
  | "tool_round_complete"
  | "state_updated"
  | "interruption"
  | "error"
  | "cost_update"
  | "turn_complete"
  | "permission_denied"
  | "session_compacted"
  | "hook_denied"
  | "hook_message"
  | "router_event"
  | "slash_command";

export interface AgentEvent<T = unknown> {
  type: AgentEventType;
  payload: T;
  timestamp: number;
}

// ── Triage ─────────────────────────────────────────────────────────

export interface TriageResult {
  /** Whether the input should be escalated to the remote model */
  escalate: boolean;
  /** Brief reasoning for the triage decision */
  reasoning: string;
  /** Direct response if the local model can answer immediately */
  directResponse?: string;
}

// ── Tool Execution ─────────────────────────────────────────────────

export interface ToolExecutionEvent {
  command: string;
  kwargs: Record<string, unknown>;
  result?: string;
  error?: string;
  durationMs?: number;
}
