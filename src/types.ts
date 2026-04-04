import type { BaseProvider } from "@sschepis/llm-wrapper";
import type { Router } from "@sschepis/swiss-army-tool";
import type { Session } from "@sschepis/as-agent";

// ── Configuration ──────────────────────────────────────────────────

export interface ObotoAgentConfig {
  /** Small, fast model for triage and summarization (e.g. Ollama, LMStudio) */
  localModel: BaseProvider;
  /** Powerful model for complex reasoning (e.g. Anthropic, OpenAI, Gemini) */
  remoteModel: BaseProvider;
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
}

// ── Event Bus ──────────────────────────────────────────────────────

export type AgentEventType =
  | "user_input"
  | "agent_thought"
  | "triage_result"
  | "tool_execution_start"
  | "tool_execution_complete"
  | "state_updated"
  | "interruption"
  | "error"
  | "turn_complete";

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
