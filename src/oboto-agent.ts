import {
  LScriptRuntime,
  MiddlewareManager,
  ExecutionCache,
  MemoryCacheBackend,
  CostTracker,
  RateLimiter,
  AgentLoop,
  type ToolDefinition,
  type ChatMessage,
  type AgentConfig,
  type ToolCall,
  type BudgetConfig,
  type ModelPricing,
} from "@sschepis/lmscript";
import type {
  StandardChatChunk,
  StandardChatResponse,
} from "@sschepis/llm-wrapper";
import { aggregateStream } from "@sschepis/llm-wrapper";
import type { Session, ConversationMessage } from "@sschepis/as-agent";
import { MessageRole } from "@sschepis/as-agent";
import type {
  ObotoAgentConfig, AgentEventType, AgentEvent, TriageResult, ProviderLike,
  AgentPhase, ToolRoundEvent,
} from "./types.js";
import { AgentEventBus } from "./event-bus.js";
import { ContextManager } from "./context-manager.js";
import { createTriageFunction } from "./triage.js";
import { createRouterTool } from "./adapters/tools.js";
import { toLmscriptProvider } from "./adapters/llm-wrapper.js";
import { toChat, createEmptySession, sessionToHistory } from "./adapters/memory.js";
import { ConversationRAG } from "./adapters/rag-integration.js";
import {
  PermissionGuard,
  SessionCompactor,
  HookIntegration,
  SlashCommandRegistry,
  AgentUsageTracker,
} from "./adapters/as-agent-features.js";
import { RouterEventBridge, isLLMRouter } from "./adapters/router-events.js";
import { UsageBridge, type UnifiedCostSummary } from "./adapters/usage-bridge.js";

type EventHandler = (event: AgentEvent) => void;

/**
 * ObotoAgent is the central orchestrator for dual-LLM agent execution.
 *
 * It binds together:
 * - llm-wrapper (LLM communication via local and remote providers / LLMRouter)
 * - lmscript (structured/schema-validated calls, agent loop, infrastructure)
 * - swiss-army-tool (tool execution via Router)
 * - as-agent (session state and conversation history)
 *
 * All interaction flows through an event-driven architecture.
 *
 * Key integration improvements (v0.2):
 * - Accepts LLMRouter for automatic failover and health tracking
 * - Uses lmscript's AgentLoop instead of a custom tool-calling loop
 * - Wires lmscript infrastructure: cache, cost tracking, rate limiting, middleware
 * - Bridges streaming via chatStream through the full lmscript stack
 */
export class ObotoAgent {
  private bus = new AgentEventBus();
  private localRuntime: LScriptRuntime;
  private remoteRuntime: LScriptRuntime;
  private localProvider: ProviderLike;
  private remoteProvider: ProviderLike;
  private contextManager: ContextManager;
  private routerTool: ToolDefinition<any, any>;
  private triageFn: ReturnType<typeof createTriageFunction>;
  private session: Session;
  private isProcessing = false;
  private interrupted = false;
  private systemPrompt: string;
  private maxIterations: number;
  private config: ObotoAgentConfig;
  private onToken?: (token: string) => void;
  private costTracker: CostTracker;
  private modelPricing?: ModelPricing;
  private rateLimiter?: RateLimiter;
  private middleware!: MiddlewareManager;
  private budget?: BudgetConfig;
  private conversationRAG?: ConversationRAG;
  private permissionGuard?: PermissionGuard;
  private sessionCompactor?: SessionCompactor;
  private hookIntegration?: HookIntegration;
  private slashCommands: SlashCommandRegistry;
  private usageTracker: AgentUsageTracker;
  private usageBridge: UsageBridge;
  private routerEventBridge: RouterEventBridge;
  private currentPhase: AgentPhase = "request";
  private shouldContinue?: (context: import("./types.js").ContinuationContext) => Promise<boolean>;
  private continuationBatchSize: number;
  private maxTotalIterations: number;

  constructor(config: ObotoAgentConfig) {
    this.config = config;
    this.localProvider = config.localModel;
    this.remoteProvider = config.remoteModel;

    // ── Build lmscript providers from llm-wrapper providers ──────────
    const localLmscript = toLmscriptProvider(config.localModel, "local");
    const remoteLmscript = toLmscriptProvider(config.remoteModel, "remote");

    // ── Build middleware manager ─────────────────────────────────────
    this.middleware = new MiddlewareManager();
    if (config.middleware) {
      for (const hooks of config.middleware) {
        this.middleware.use(hooks);
      }
    }

    // ── Build cost tracker (shared across runtimes for unified reporting) ─
    // Always create a CostTracker so token counts are tracked even without pricing.
    // Pricing is passed to getTotalCost() when available.
    this.costTracker = new CostTracker();
    this.modelPricing = config.modelPricing;

    // ── Build local runtime (lightweight: cache for triage, no rate limit) ─
    // ExecutionCache takes only a CacheBackend; TTL is handled per-entry
    // by the MemoryCacheBackend.set(key, value, ttlMs) call
    const localCache = config.triageCacheTtlMs
      ? new ExecutionCache(new MemoryCacheBackend())
      : undefined;

    this.localRuntime = new LScriptRuntime({
      provider: localLmscript,
      defaultTemperature: 0.1,
      cache: localCache,
      costTracker: this.costTracker,
    });

    // ── Build remote runtime (full infrastructure) ──────────────────
    const remoteCache = config.cacheTtlMs
      ? new ExecutionCache(new MemoryCacheBackend())
      : undefined;

    const rateLimiter = config.rateLimit
      ? new RateLimiter(config.rateLimit)
      : undefined;
    this.rateLimiter = rateLimiter;
    this.budget = config.budget;

    this.remoteRuntime = new LScriptRuntime({
      provider: remoteLmscript,
      middleware: this.middleware,
      cache: remoteCache,
      costTracker: this.costTracker,
      budget: config.budget,
      rateLimiter,
    });

    // ── Session and context ─────────────────────────────────────────
    this.session = config.session ?? createEmptySession();
    this.systemPrompt = config.systemPrompt ?? "You are a helpful AI assistant with access to tools.";
    this.maxIterations = config.maxIterations ?? 10;
    this.onToken = config.onToken;

    // Adaptive iteration control
    this.shouldContinue = config.shouldContinue;
    this.continuationBatchSize = config.continuationBatchSize ?? 10;
    this.maxTotalIterations = config.maxTotalIterations ?? 100;

    this.contextManager = new ContextManager(
      this.localRuntime,
      config.localModelName,
      config.maxContextTokens ?? 8192
    );

    // ── Tool layer ──────────────────────────────────────────────────
    // Apply swiss-army-tool middleware if configured
    if (config.toolMiddleware) {
      for (const mw of config.toolMiddleware) {
        config.router.use(mw);
      }
    }

    this.routerTool = createRouterTool(config.router);
    this.triageFn = createTriageFunction(config.localModelName);

    // ── RAG pipeline (optional) ─────────────────────────────────────
    if (config.embeddingProvider) {
      this.conversationRAG = new ConversationRAG(this.remoteRuntime, {
        embeddingProvider: config.embeddingProvider,
        vectorStore: config.vectorStore,
        topK: config.ragTopK,
        minScore: config.ragMinScore,
        embeddingModel: config.ragEmbeddingModel,
        autoIndex: config.ragAutoIndex,
        indexToolResults: config.ragIndexToolResults,
        formatContext: config.ragFormatContext,
      });
    }

    // ── as-agent features ───────────────────────────────────────────
    // Permission guard (optional)
    if (config.permissionPolicy) {
      this.permissionGuard = new PermissionGuard(
        config.permissionPolicy,
        config.permissionPrompter ?? null,
        this.bus,
      );
    }

    // Session compaction (optional)
    if (config.compactionConfig) {
      this.sessionCompactor = new SessionCompactor(this.bus, config.compactionConfig);
    }

    // Hook runner (optional)
    if (config.hookRunner) {
      this.hookIntegration = new HookIntegration(config.hookRunner, this.bus);
    }

    // Slash command registry (always available, Wasm runtime optional)
    this.slashCommands = new SlashCommandRegistry(config.agentRuntime);

    // Usage tracker (always available)
    this.usageTracker = new AgentUsageTracker();

    // Usage bridge: unifies as-agent UsageTracker and lmscript CostTracker
    this.usageBridge = new UsageBridge(this.usageTracker, this.costTracker);

    // ── Router event bridge (auto-attach LLMRouters) ────────────────
    this.routerEventBridge = new RouterEventBridge(this.bus);
    if (isLLMRouter(config.localModel)) {
      this.routerEventBridge.attach(config.localModel, "local");
    }
    if (isLLMRouter(config.remoteModel)) {
      this.routerEventBridge.attach(config.remoteModel, "remote");
    }

    // Push system prompt into context
    this.contextManager.push({
      role: "system",
      content: this.systemPrompt,
    });

    // Seed context manager from existing session messages so triage
    // and execution paths see the full conversation history.
    if (this.session.messages.length > 0) {
      this.contextManager.pushAll(sessionToHistory(this.session));
    }
  }

  // ── Public API ─────────────────────────────────────────────────────

  /** Subscribe to agent events. Returns an unsubscribe function. */
  on(type: AgentEventType, handler: EventHandler): () => void {
    return this.bus.on(type, handler);
  }

  /** Subscribe to an event for a single emission. */
  once(type: AgentEventType, handler: EventHandler): () => void {
    return this.bus.once(type, handler);
  }

  /** Submit user input to the agent. Triggers the execution loop. */
  async submitInput(text: string): Promise<void> {
    if (this.isProcessing) {
      this.interrupt(text);
      return;
    }

    // Check for slash commands first
    const parsedCmd = this.slashCommands.parseCommand(text);
    if (parsedCmd) {
      const result = await this.slashCommands.executeCustomCommand(parsedCmd.name, parsedCmd.args);
      if (result !== null) {
        this.bus.emit("slash_command", {
          command: parsedCmd.name,
          args: parsedCmd.args,
          result,
        });
        return;
      }
      // If not a custom command, fall through to normal processing
      // (the Wasm runtime may handle it, or it's treated as regular input)
    }

    this.isProcessing = true;
    this.interrupted = false;

    try {
      await this.executionLoop(text);

      // End the usage tracker turn (via bridge)
      this.usageBridge.endTurn();

      // Auto-compact session if configured
      if (this.sessionCompactor) {
        const result = this.sessionCompactor.compactIfNeeded(this.session);
        if (result && result.removedMessageCount > 0) {
          this.session = result.compactedSession;
        }
      }
    } catch (err) {
      this.bus.emit("error", {
        message: err instanceof Error ? err.message : String(err),
        error: err,
      });
    } finally {
      this.isProcessing = false;
    }
  }

  /**
   * Interrupt the current execution loop.
   * Optionally inject new directives into the context.
   */
  async interrupt(newDirectives?: string): Promise<void> {
    this.interrupted = true;
    this.bus.emit("interruption", { newDirectives });

    if (newDirectives) {
      const msg: ConversationMessage = {
        role: MessageRole.User,
        blocks: [{ kind: "text", text: `[INTERRUPTION] ${newDirectives}` }],
      };
      await this.recordMessage(msg);
      this.bus.emit("state_updated", { reason: "interruption" });
    }
  }

  /** Get the current session state. */
  getSession(): Session {
    return this.session;
  }

  /** Whether the agent is currently processing input. */
  get processing(): boolean {
    return this.isProcessing;
  }

  /** Get cost tracking summary (if cost tracking is enabled). */
  getCostSummary(): {
    totalCost: number;
    totalTokens: number;
    byFunction: Record<string, { calls: number; totalTokens: number; promptTokens: number; completionTokens: number }>;
  } | undefined {
    if (!this.costTracker) return undefined;

    const totalTokens = this.costTracker.getTotalTokens();
    const totalCost = this.costTracker.getTotalCost(this.modelPricing);

    // Convert Map to plain object for easier consumption
    const usageMap = this.costTracker.getUsageByFunction();
    const byFunction: Record<string, { calls: number; totalTokens: number; promptTokens: number; completionTokens: number }> = {};
    for (const [fnName, entry] of usageMap) {
      byFunction[fnName] = entry;
    }

    return { totalCost, totalTokens, byFunction };
  }

  /**
   * Get unified cost summary combining both as-agent and lmscript tracking.
   * Uses the UsageBridge to provide a single view of all token/cost data.
   */
  getUnifiedCostSummary(
    asPricing?: import("@sschepis/as-agent").ModelPricing,
  ): UnifiedCostSummary {
    return this.usageBridge.getCostSummary(this.modelPricing, asPricing);
  }

  /** Get the usage bridge for direct access to unified tracking. */
  getUsageBridge(): UsageBridge {
    return this.usageBridge;
  }

  /** Remove all event listeners and detach router event subscriptions. */
  removeAllListeners(): void {
    this.bus.removeAllListeners();
    this.routerEventBridge.detachAll();
  }

  /** Sync session and repopulate context manager (for reuse across turns). */
  async syncSession(session: Session): Promise<void> {
    this.session = session;
    this.contextManager.clear();
    await this.contextManager.push({ role: "system", content: this.systemPrompt });
    if (session.messages.length > 0) {
      await this.contextManager.pushAll(sessionToHistory(session));
    }
  }

  /** Update the streaming token callback between turns. */
  setOnToken(callback: ((token: string) => void) | undefined): void {
    this.onToken = callback;
  }

  /** Get the ConversationRAG instance (if RAG is enabled). */
  getConversationRAG(): ConversationRAG | undefined {
    return this.conversationRAG;
  }

  /** Get the slash command registry. */
  getSlashCommands(): SlashCommandRegistry {
    return this.slashCommands;
  }

  /** Get the as-agent usage tracker. */
  getUsageTracker(): AgentUsageTracker {
    return this.usageTracker;
  }

  /** Get the permission guard (if permissions are configured). */
  getPermissionGuard(): PermissionGuard | undefined {
    return this.permissionGuard;
  }

  /** Get the session compactor (if compaction is configured). */
  getSessionCompactor(): SessionCompactor | undefined {
    return this.sessionCompactor;
  }

  /** Get the router event bridge for observability into LLMRouter health. */
  getRouterEventBridge(): RouterEventBridge {
    return this.routerEventBridge;
  }

  /**
   * Manually compact the session. Returns null if compaction is not configured.
   */
  compactSession(): import("@sschepis/as-agent").CompactionResult | null {
    if (!this.sessionCompactor) return null;
    const result = this.sessionCompactor.compact(this.session);
    if (result.removedMessageCount > 0) {
      this.session = result.compactedSession;
    }
    return result;
  }

  /**
   * Retrieve relevant past context for a query via the RAG pipeline.
   * Returns undefined if RAG is not configured.
   */
  async retrieveContext(query: string): Promise<string | undefined> {
    if (!this.conversationRAG) return undefined;
    const { context } = await this.conversationRAG.retrieve(query);
    return context || undefined;
  }

  // ── Conversational helpers ──────────────────────────────────────────

  /** Emit a phase transition event with a human-readable message. */
  private emitPhase(phase: AgentPhase, message: string): void {
    this.currentPhase = phase;
    this.bus.emit("phase", { phase, message });
  }

  /**
   * Build a human-readable narrative for a batch of tool executions.
   * Uses both command name and kwargs to produce accurate descriptions.
   * E.g. "Just read file data, and edited files. Sending results back to AI for next steps…"
   */
  private buildToolRoundNarrative(
    tools: Array<{ command: string; success: boolean; kwargs?: Record<string, unknown> }>
  ): string {
    if (tools.length === 0) return "No tools executed.";

    // Classify each tool call by examining both the command and its kwargs
    const verbs = new Map<string, string[]>();
    for (const t of tools) {
      const cmd = t.command.toLowerCase();
      const kw = t.kwargs || {};
      const kwStr = JSON.stringify(kw).toLowerCase();

      let verb: string;
      // Check kwargs for more specific classification
      if (kwStr.includes('"cmd"') && (kwStr.includes("cat ") || kwStr.includes("head ") || kwStr.includes("tail "))) {
        verb = "read file data";
      } else if (kwStr.includes('"cmd"') && (kwStr.includes("ls ") || kwStr.includes("find ") || kwStr.includes("tree "))) {
        verb = "listed files";
      } else if (kwStr.includes('"cmd"') && (kwStr.includes("grep ") || kwStr.includes("rg ") || kwStr.includes("ag "))) {
        verb = "searched for information";
      } else if (cmd.includes("read") || cmd.includes("get_file") || cmd.includes("view")) {
        verb = "read file data";
      } else if (cmd.includes("write") || cmd.includes("edit") || cmd.includes("patch") || cmd.includes("update")) {
        verb = "edited files";
      } else if (cmd.includes("search") || cmd.includes("grep") || cmd.includes("find") || cmd.includes("glob")) {
        verb = "searched for information";
      } else if (cmd.includes("list") || cmd.includes("ls")) {
        verb = "listed items";
      } else if (cmd.includes("run") || cmd.includes("exec") || cmd.includes("bash") || cmd.includes("shell")) {
        verb = "ran a command";
      } else if (cmd.includes("browse") || cmd.includes("web") || cmd.includes("fetch")) {
        verb = "browsed the web";
      } else if (cmd.includes("surface")) {
        verb = cmd.includes("read") ? "read surface data" : "accessed surfaces";
      } else if (cmd.includes("help")) {
        verb = "checked available tools";
      } else {
        verb = `used ${t.command}`;
      }
      if (!verbs.has(verb)) verbs.set(verb, []);
      verbs.get(verb)!.push(t.command);
    }

    const parts = Array.from(verbs.keys());
    let narrative: string;
    if (parts.length === 1) {
      narrative = `Just ${parts[0]}.`;
    } else if (parts.length === 2) {
      narrative = `Just ${parts[0]}, and ${parts[1]}.`;
    } else {
      const last = parts.pop()!;
      narrative = `Just ${parts.join(", ")}, and ${last}.`;
    }

    const errors = tools.filter(t => !t.success);
    if (errors.length > 0) {
      const errNames = errors.map(e => e.command).join(", ");
      narrative += ` (${errors.length} error${errors.length > 1 ? "s" : ""}: ${errNames})`;
    }

    narrative += " Sending results back to AI for next steps…";
    return narrative;
  }

  // ── Internal ───────────────────────────────────────────────────────

  /**
   * Record a message in the session, context manager, and optionally RAG index.
   * Centralizes message recording to ensure RAG indexing stays in sync.
   */
  private async recordMessage(msg: ConversationMessage): Promise<void> {
    this.session.messages.push(msg);
    await this.contextManager.push(toChat(msg));

    // Auto-index for RAG if enabled
    if (this.conversationRAG) {
      // Fire and forget — RAG indexing should not block the main loop
      this.conversationRAG.indexMessage(msg, this.session.messages.length - 1).catch((err) => {
        console.warn("[ObotoAgent] RAG indexing failed:", err instanceof Error ? err.message : err);
      });
    }
  }

  /**
   * Record a tool execution result in the RAG index.
   */
  private recordToolResult(command: string, kwargs: Record<string, unknown>, result: string): void {
    if (this.conversationRAG) {
      this.conversationRAG.indexToolResult(command, kwargs, result).catch((err) => {
        console.warn("[ObotoAgent] RAG tool indexing failed:", err instanceof Error ? err.message : err);
      });
    }
  }

  private async executionLoop(userInput: string): Promise<void> {
    // ── Phase: Request ──
    this.emitPhase("request", `Processing: ${userInput.substring(0, 80)}${userInput.length > 80 ? "…" : ""}`);

    // 1. Emit user_input and record in session + context + RAG
    this.bus.emit("user_input", { text: userInput });

    const userMsg: ConversationMessage = {
      role: MessageRole.User,
      blocks: [{ kind: "text", text: userInput }],
    };
    await this.recordMessage(userMsg);
    this.bus.emit("state_updated", { reason: "user_input" });

    // ── Phase: Planning ──
    this.emitPhase("planning", "Building context and preparing tools…");

    // 1b. Optionally augment context with RAG-retrieved past conversation
    if (this.conversationRAG) {
      try {
        const { context } = await this.conversationRAG.retrieve(userInput);
        if (context) {
          await this.contextManager.push({
            role: "system",
            content: context,
          });
          this.bus.emit("agent_thought", {
            text: "Retrieved relevant past context via RAG.",
            model: "system",
          });
        }
      } catch (err) {
        console.warn("[ObotoAgent] RAG retrieval failed:", err instanceof Error ? err.message : err);
      }
    }

    // ── Phase: Precheck (Triage) ──
    // StreamController's ActivityTracker handles heartbeat automatically
    // when phaseStart is called, so we don't need our own timer.
    this.emitPhase("precheck", "Checking if direct answer is possible…");

    const triageResult = await this.triage(userInput);
    this.bus.emit("triage_result", triageResult);

    if (this.interrupted) return;

    // 3. If local can handle directly, emit and return
    if (!triageResult.escalate && triageResult.directResponse) {
      const response = triageResult.directResponse;
      this.emitPhase("complete", "Answered directly — no tools needed.");
      this.bus.emit("agent_thought", { text: response, model: "local" });

      const assistantMsg: ConversationMessage = {
        role: MessageRole.Assistant,
        blocks: [{ kind: "text", text: response }],
      };
      await this.recordMessage(assistantMsg);
      this.bus.emit("state_updated", { reason: "assistant_response" });
      this.bus.emit("turn_complete", { model: "local", escalated: false });
      return;
    }

    // 4. Escalate to remote model with tool access
    // Note: triage_result event already carries the reasoning for the provider to display.
    // We only emit the phase transition here — no duplicate agent_thought.
    if (triageResult.escalate) {
      this.emitPhase("thinking", "Entering agent loop with remote model…");
    } else {
      this.emitPhase("thinking", "This requires tools and deeper reasoning — entering agent loop.");
    }

    const modelName = triageResult.escalate
      ? this.config.remoteModelName
      : this.config.localModelName;

    const runtime = triageResult.escalate
      ? this.remoteRuntime
      : this.localRuntime;

    if (this.onToken) {
      await this.executeWithStreaming(runtime, modelName, userInput);
    } else {
      await this.executeWithAgentLoop(runtime, modelName, userInput);
    }
  }

  private async triage(userInput: string): Promise<TriageResult> {
    const recentMessages = this.contextManager.getMessages().slice(-5);
    const recentContext = recentMessages
      .map((m) => {
        const text = typeof m.content === "string" ? m.content : "[complex content]";
        return `${m.role}: ${text}`;
      })
      .join("\n");

    const result = await this.localRuntime.execute(this.triageFn, {
      userInput,
      recentContext,
      availableTools: this.routerTool.description,
    });

    return result.data;
  }

  /**
   * Execute using lmscript's AgentLoop for iterative tool calling.
   * This replaces the old custom tool loop with lmscript's battle-tested implementation.
   *
   * Benefits:
   * - Schema validation on final output
   * - Budget checking via CostTracker
   * - Rate limiting via RateLimiter
   * - Middleware lifecycle hooks
   * - Automatic retry with backoff
   */
  private async executeWithAgentLoop(
    runtime: LScriptRuntime,
    modelName: string,
    userInput: string
  ): Promise<void> {
    const { z } = await import("zod");

    this.emitPhase("thinking", `Turn 1/${this.maxIterations}: Analyzing request…`);

    const agentFn = {
      name: "agent-task",
      model: modelName,
      system: this.systemPrompt,
      prompt: (input: string) => {
        const contextMessages = this.contextManager.getMessages();
        const contextStr = contextMessages
          .filter(m => m.role !== "system")
          .map(m => {
            const text = typeof m.content === "string" ? m.content : "[complex content]";
            return `${m.role}: ${text}`;
          })
          .join("\n");

        return contextStr ? `${contextStr}\n\nuser: ${input}` : input;
      },
      schema: z.object({
        response: z.string().describe("The assistant's response to the user"),
        reasoning: z.string().optional().describe("Internal reasoning about the approach taken"),
      }),
      tools: [this.routerTool],
      temperature: 0.7,
      maxRetries: 1,
    };

    // Track tool calls per iteration for narrative building
    let iterationTools: Array<{ command: string; success: boolean; kwargs?: Record<string, unknown> }> = [];
    let totalToolCalls = 0;

    const agentConfig: AgentConfig = {
      maxIterations: this.maxIterations,
      onToolCall: (tc: ToolCall) => {
        const command = typeof tc.arguments === "object" && tc.arguments !== null
          ? (tc.arguments as Record<string, unknown>).command ?? tc.name
          : tc.name;
        const kwargs = typeof tc.arguments === "object" && tc.arguments !== null
          ? (tc.arguments as Record<string, unknown>).kwargs ?? {}
          : {};

        const resultStr = typeof tc.result === "string" ? tc.result : JSON.stringify(tc.result);
        const isError = resultStr.startsWith("Error:");

        this.emitPhase("tools", `Running tool: ${String(command)}`);
        this.bus.emit("tool_execution_start", { command, kwargs });
        this.bus.emit("tool_execution_complete", { command, kwargs, result: resultStr });

        iterationTools.push({ command: String(command), success: !isError, kwargs: kwargs as Record<string, unknown> });
        totalToolCalls++;

        this.recordToolResult(String(command), kwargs as Record<string, unknown>, resultStr);

        // Persist tool interactions in session + contextManager for cross-turn memory.
        // onToolCall may be synchronous in lmscript's API, so use fire-and-forget.
        const toolCallId = `call_${Date.now()}_${totalToolCalls}`;
        this.recordMessage({
          role: MessageRole.Assistant,
          blocks: [{ kind: "tool_use", id: toolCallId, name: String(command), input: JSON.stringify(kwargs).substring(0, 500) }],
        }).catch(() => {});
        this.recordMessage({
          role: MessageRole.Tool,
          blocks: [{ kind: "tool_result", toolUseId: toolCallId, toolName: String(command), output: resultStr.substring(0, 2000), isError }],
        }).catch(() => {});
      },
      onIteration: (iteration: number, response: string) => {
        // Emit tool round narrative if tools were called this iteration
        if (iterationTools.length > 0) {
          const narrative = this.buildToolRoundNarrative(iterationTools);
          const roundEvent: ToolRoundEvent = {
            iteration,
            tools: iterationTools,
            totalToolCalls,
            narrative,
          };
          this.bus.emit("tool_round_complete", roundEvent);
          iterationTools = [];
        }

        // Forward AI text
        if (response) {
          this.bus.emit("agent_thought", {
            text: response,
            model: modelName,
            iteration,
          });
        }

        // Announce next iteration
        this.emitPhase("thinking", `Turn ${iteration + 1}/${this.maxIterations}: Analyzing results…`);

        if (this.interrupted) return false;
      },
    };

    const agentLoop = new AgentLoop(runtime, agentConfig);
    const result = await agentLoop.run(agentFn, userInput);

    // Emit final tool round if pending
    if (iterationTools.length > 0) {
      const narrative = this.buildToolRoundNarrative(iterationTools);
      this.bus.emit("tool_round_complete", {
        iteration: result.iterations,
        tools: iterationTools,
        totalToolCalls,
        narrative,
      });
    }

    // ── Phase: Memory ──
    this.emitPhase("memory", "Recording interaction…");

    const responseText = result.data.response;
    const assistantMsg: ConversationMessage = {
      role: MessageRole.Assistant,
      blocks: [{ kind: "text", text: responseText }],
    };
    await this.recordMessage(assistantMsg);
    this.bus.emit("state_updated", { reason: "assistant_response" });

    // ── Phase: Complete ──
    this.emitPhase("complete", "Response ready.");
    this.bus.emit("turn_complete", {
      model: modelName,
      escalated: true,
      iterations: result.iterations,
      toolCalls: result.toolCalls.length,
      usage: result.usage,
    });
  }

  /**
   * Execute with streaming token emission.
   * Uses the raw llm-wrapper provider for streaming, combined with
   * manual tool calling (since streaming + structured agent loops are complex).
   *
   * This preserves real-time token delivery while still leveraging
   * the lmscript infrastructure:
   * - Rate limiting (acquire/reportTokens per call)
   * - Cost tracking (trackUsage per call)
   * - Budget checking (checkBudget before each call)
   * - Middleware lifecycle hooks (onBeforeExecute/onComplete per turn)
   */
  private async executeWithStreaming(
    _runtime: LScriptRuntime,
    modelName: string,
    _userInput: string
  ): Promise<void> {
    const { zodToJsonSchema } = await import("zod-to-json-schema");

    const provider = modelName === this.config.remoteModelName
      ? this.remoteProvider
      : this.localProvider;

    const contextMessages = this.contextManager.getMessages();

    const messages: import("@sschepis/llm-wrapper").Message[] = contextMessages.map((m) => ({
      role: m.role,
      content:
        typeof m.content === "string"
          ? m.content
          : (m.content as Array<{ type: string; text?: string }>)
              .filter((b) => b.type === "text")
              .map((b) => b.text ?? "")
              .join("\n"),
    }));

    const tool = this.routerTool;
    const parametersSchema = tool.parameters
      ? (zodToJsonSchema(tool.parameters, { target: "openApi3" }) as Record<string, unknown>)
      : { type: "object", properties: {} };
    const tools: import("@sschepis/llm-wrapper").ToolDefinition[] = [
      {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: parametersSchema,
        },
      },
    ];

    let totalToolCalls = 0;
    const callHistory: string[] = [];
    const toolsUsedSet = new Set<string>();
    // Track consecutive duplicate patterns for doom loop detection
    const consecutiveDupes = new Map<string, number>();
    let doomLoopRedirected = false;
    let effectiveMaxIterations = this.maxIterations;
    const turnStartTime = Date.now();
    const totalUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    const syntheticCtx = {
      fn: { name: "streaming-turn", model: modelName, system: this.systemPrompt, prompt: () => "", schema: {} as any },
      input: _userInput,
      messages: messages as any,
      attempt: 1,
      startTime: turnStartTime,
    };
    await this.middleware.runBeforeExecute(syntheticCtx);

    try {
      for (let iteration = 1; iteration <= effectiveMaxIterations; iteration++) {
        if (this.interrupted) {
          this.emitPhase("cancel", "Interrupted by user.");
          break;
        }

        // ── Phase: Thinking with iteration context ──
        this.emitPhase(
          "thinking",
          `Turn ${iteration}/${effectiveMaxIterations}: ${iteration === 1 ? "Analyzing request…" : "Analyzing results…"}`
        );

        // ── Budget check ──
        if (this.budget) {
          this.costTracker.checkBudget(this.budget);
        }

        await this.rateLimiter?.acquire();

        let isLastIteration = iteration === effectiveMaxIterations;

        // ── Adaptive continuation: ask shouldContinue before hitting the wall ──
        if (isLastIteration && this.shouldContinue && effectiveMaxIterations < this.maxTotalIterations) {
          const lastContent = messages.filter(m => m.role === "assistant").pop()?.content;
          try {
            const shouldGo = await this.shouldContinue({
              currentIteration: iteration,
              currentLimit: effectiveMaxIterations,
              totalToolCalls,
              uniqueToolsUsed: toolsUsedSet.size,
              toolsUsed: [...toolsUsedSet],
              doomDetected: doomLoopRedirected,
              lastContent: typeof lastContent === "string" ? lastContent : "",
              usage: { ...totalUsage },
            });
            if (shouldGo) {
              const extension = Math.min(
                this.continuationBatchSize,
                this.maxTotalIterations - effectiveMaxIterations,
              );
              effectiveMaxIterations += extension;
              isLastIteration = false;
              this.emitPhase(
                "continuation",
                `Extending: ${extension} more iterations granted (${effectiveMaxIterations} total)`
              );
            }
          } catch (err) {
            // shouldContinue failure is non-fatal — proceed with hard stop
            console.warn("[ObotoAgent] shouldContinue callback failed:", err);
          }
        }

        if (isLastIteration) {
          this.emitPhase("continuation", "Final iteration — synthesizing response…");
          messages.push({
            role: "user",
            content:
              "You have used all available tool iterations. Please provide your final response now based on what you have gathered so far. Do not call any more tools.",
          });
        }

        const params: import("@sschepis/llm-wrapper").StandardChatParams = {
          model: modelName,
          messages: [...messages],
          temperature: 0.7,
          ...(isLastIteration
            ? {}
            : { tools, tool_choice: "auto" as const }),
        };

        let response: StandardChatResponse;
        try {
          response = await this.streamAndAggregate(provider, params);
        } catch (err) {
          this.emitPhase("error", `LLM call failed: ${err instanceof Error ? err.message : String(err)}`);
          await this.middleware.runError(
            syntheticCtx,
            err instanceof Error ? err : new Error(String(err))
          );
          throw err;
        }

        // ── Usage tracking ──
        const usage = response?.usage;
        if (usage) {
          const promptTokens = usage.prompt_tokens ?? 0;
          const completionTokens = usage.completion_tokens ?? 0;
          const usageTotal = usage.total_tokens ?? (promptTokens + completionTokens);

          totalUsage.promptTokens += promptTokens;
          totalUsage.completionTokens += completionTokens;
          totalUsage.totalTokens += usageTotal;

          this.rateLimiter?.reportTokens(usageTotal);
          this.usageBridge.recordFromLmscript(modelName, {
            promptTokens,
            completionTokens,
            totalTokens: usageTotal,
          });

          this.bus.emit("cost_update", {
            iteration,
            totalTokens: this.costTracker.getTotalTokens(),
            totalCost: this.costTracker.getTotalCost(this.modelPricing),
          });
        }

        const choice = response?.choices?.[0];
        const content = (choice?.message?.content as string) ?? "";
        const toolCalls = choice?.message?.tool_calls;

        // Forward AI reasoning text
        if (content) {
          this.bus.emit("agent_thought", {
            text: content,
            model: modelName,
            iteration,
          });
        }

        // ── No tool calls → final response ──
        if (!toolCalls || toolCalls.length === 0) {
          // Handle empty responses
          if (!content) {
            this.bus.emit("agent_thought", {
              text: `Empty response from AI — iteration ${iteration}`,
              model: "system",
            });
            if (iteration < this.maxIterations) continue;
          }

          this.emitPhase("memory", "Recording interaction…");

          const assistantMsg: ConversationMessage = {
            role: MessageRole.Assistant,
            blocks: [{ kind: "text", text: content }],
          };
          await this.recordMessage(assistantMsg);
          this.bus.emit("state_updated", { reason: "assistant_response" });

          this.emitPhase("complete", "Response ready.");
          this.bus.emit("turn_complete", {
            model: modelName,
            escalated: true,
            iterations: iteration,
            toolCalls: totalToolCalls,
            usage: totalUsage,
          });

          await this.middleware.runComplete(syntheticCtx, {
            data: content,
            raw: content,
            usage: totalUsage,
          } as any);
          return;
        }

        // ── Phase: Tools ──
        this.emitPhase("tools", `Executing ${toolCalls.length} tool(s)…`);

        messages.push({
          role: "assistant",
          content: content || null,
          tool_calls: toolCalls,
        });

        // Persist assistant tool-calling intent in session + contextManager
        // so it survives across turns (the local `messages` array is ephemeral).
        const toolBlocks: import("@sschepis/as-agent").ContentBlock[] = [];
        if (content) {
          toolBlocks.push({ kind: "text", text: content });
        }
        for (const tc of toolCalls) {
          let parsedTcArgs: Record<string, unknown> = {};
          try { parsedTcArgs = JSON.parse(tc.function.arguments); } catch {}
          const tcCommand = (parsedTcArgs.command as string) ?? tc.function.name;
          toolBlocks.push({
            kind: "tool_use",
            id: tc.id,
            name: tcCommand,
            input: tc.function.arguments?.substring(0, 500) || "{}",
          });
        }
        await this.recordMessage({
          role: MessageRole.Assistant,
          blocks: toolBlocks,
        });

        // Track tools this round for narrative
        const roundTools: Array<{ command: string; success: boolean; kwargs?: Record<string, unknown> }> = [];

        for (let ti = 0; ti < toolCalls.length; ti++) {
          const tc = toolCalls[ti];
          if (this.interrupted) break;

          let args: Record<string, unknown>;
          try {
            args = JSON.parse(tc.function.arguments);
          } catch {
            args = {};
          }

          const command = (args.command as string) ?? tc.function.name;
          const kwargs = (args.kwargs as Record<string, unknown>) ?? {};
          const toolInputStr = JSON.stringify({ command, kwargs });

          // ── Permission check ──
          if (this.permissionGuard) {
            const outcome = this.permissionGuard.checkPermission(command, toolInputStr);
            if (outcome.kind === "deny") {
              messages.push({
                role: "tool",
                tool_call_id: tc.id,
                content: `Permission denied for tool "${command}": ${outcome.reason ?? "denied by policy"}`,
              });
              roundTools.push({ command, success: false });
              totalToolCalls++;
              continue;
            }
          }

          // ── Pre-tool-use hooks ──
          if (this.hookIntegration) {
            const hookResult = this.hookIntegration.runPreToolUse(command, toolInputStr);
            if (hookResult.denied) {
              messages.push({
                role: "tool",
                tool_call_id: tc.id,
                content: `Tool "${command}" blocked by pre-use hook: ${hookResult.messages.join("; ")}`,
              });
              roundTools.push({ command, success: false });
              totalToolCalls++;
              continue;
            }
          }

          // ── Duplicate detection + doom loop ──
          const callSig = JSON.stringify({ command, kwargs });
          const dupeCount = callHistory.filter((s) => s === callSig).length;
          callHistory.push(callSig);

          // Track consecutive dupes for doom loop detection
          const prevConsec = consecutiveDupes.get(command) ?? 0;
          consecutiveDupes.set(command, prevConsec + 1);

          if (dupeCount >= 2) {
            // ── Doom loop detection ──
            if (prevConsec >= 3 && !doomLoopRedirected) {
              doomLoopRedirected = true;
              this.emitPhase("doom", `Doom loop detected: repeated calls to "${command}"`);
              this.bus.emit("doom_loop", {
                reason: `Repeated calls to "${command}"`,
                command,
                count: prevConsec + 1,
                redirected: true,
              });
              // Inject a redirect message to break the pattern
              messages.push({
                role: "tool",
                tool_call_id: tc.id,
                content: `STOP: You have been calling "${command}" repeatedly with the same arguments ${prevConsec + 1} times. `
                  + `This is a doom loop. You MUST take a different approach. `
                  + `Summarize what you know so far and either try a completely different strategy or provide your best answer.`,
              });
            } else if (doomLoopRedirected && prevConsec >= 5) {
              // Persistent doom loop — force termination
              this.emitPhase("doom", `Persistent doom loop — terminating.`);
              this.bus.emit("doom_loop", {
                reason: `Persistent doom loop on "${command}"`,
                command,
                count: prevConsec + 1,
                redirected: false,
              });
              // Break out of the tool loop and the iteration loop
              roundTools.push({ command, success: false });
              totalToolCalls++;

              // Emit tool round narrative before breaking
              if (roundTools.length > 0) {
                const narrative = this.buildToolRoundNarrative(roundTools);
                this.bus.emit("tool_round_complete", {
                  iteration,
                  tools: roundTools,
                  totalToolCalls,
                  narrative,
                } as ToolRoundEvent);
              }

              // Force final response
              this.emitPhase("continuation", "Synthesizing response after doom loop…");
              const assistantMsg: ConversationMessage = {
                role: MessageRole.Assistant,
                blocks: [{ kind: "text", text: content || "I encountered a repeating pattern and am unable to make further progress. Here is what I have so far." }],
              };
              await this.recordMessage(assistantMsg);
              this.bus.emit("state_updated", { reason: "doom_loop" });
              this.emitPhase("complete", "Response ready.");
              this.bus.emit("turn_complete", {
                model: modelName,
                escalated: true,
                iterations: iteration,
                toolCalls: totalToolCalls,
                usage: totalUsage,
              });
              await this.middleware.runComplete(syntheticCtx, {
                data: content || "doom_loop_terminated",
                raw: content,
                usage: totalUsage,
              } as any);
              return;
            } else {
              messages.push({
                role: "tool",
                tool_call_id: tc.id,
                content: `You already called "${command}" with these arguments ${dupeCount} time(s). Use the data you already have.`,
              });
            }

            roundTools.push({ command, success: false });
            totalToolCalls++;
            continue;
          } else {
            // Reset consecutive counter for this command on a novel call
            consecutiveDupes.set(command, 0);
          }

          // ── Tool status: "Running tool 1/3: read_file" ──
          this.bus.emit("tool_execution_start", { command, kwargs, index: ti, total: toolCalls.length });

          let result: string;
          let isError = false;
          try {
            result = await tool.execute(args);
          } catch (err) {
            result = `Error: ${err instanceof Error ? err.message : String(err)}`;
            isError = true;
          }

          const resultStr = typeof result === "string" ? result : JSON.stringify(result);
          const truncated =
            resultStr.length > 8000
              ? resultStr.slice(0, 8000) +
                `\n\n[... truncated ${resultStr.length - 8000} characters.]`
              : resultStr;

          if (this.hookIntegration) {
            this.hookIntegration.runPostToolUse(command, toolInputStr, truncated, isError);
          }

          this.bus.emit("tool_execution_complete", { command, kwargs, result: truncated, error: isError ? truncated : undefined });
          this.recordToolResult(command, kwargs, truncated);
          roundTools.push({ command, success: !isError, kwargs });
          totalToolCalls++;
          toolsUsedSet.add(command);

          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: truncated,
          });

          // Persist tool result in session + contextManager for cross-turn memory
          await this.recordMessage({
            role: MessageRole.Tool,
            blocks: [{
              kind: "tool_result",
              toolUseId: tc.id,
              toolName: command,
              output: truncated.substring(0, 2000),
              isError,
            }],
          });
        }

        // ── Tool round narrative ──
        if (roundTools.length > 0) {
          const narrative = this.buildToolRoundNarrative(roundTools);
          this.bus.emit("tool_round_complete", {
            iteration,
            tools: roundTools,
            totalToolCalls,
            narrative,
          } as ToolRoundEvent);
        }
      }

      // ── Exhausted iterations fallback ──
      // If we reach here, the for-loop completed without the LLM producing
      // a final tool-free response.  The last iteration already injected
      // a "provide your final response" prompt, so the LLM *should* have
      // answered — but if it didn't, record what we have.
      this.emitPhase("continuation", "All iterations used — finalizing…");

      // Grab the last assistant content from messages as the fallback
      let fallbackText = "";
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === "assistant" && typeof messages[i].content === "string" && messages[i].content) {
          fallbackText = messages[i].content as string;
          break;
        }
      }
      if (!fallbackText) {
        fallbackText = "I wasn't able to complete the task within the available iterations. Please let me know how you'd like to proceed.";
      }

      const fallbackMsg: ConversationMessage = {
        role: MessageRole.Assistant,
        blocks: [{ kind: "text", text: fallbackText }],
      };
      await this.recordMessage(fallbackMsg);
      this.bus.emit("state_updated", { reason: "max_iterations" });

      this.emitPhase("complete", "Response ready.");
      this.bus.emit("turn_complete", {
        model: modelName,
        escalated: true,
        iterations: effectiveMaxIterations,
        toolCalls: totalToolCalls,
        usage: totalUsage,
      });

      await this.middleware.runComplete(syntheticCtx, {
        data: "max_iterations_reached",
        raw: "",
        usage: totalUsage,
      } as any);

    } catch (err) {
      this.emitPhase("error", `Error: ${err instanceof Error ? err.message : String(err)}`);
      if (!(err instanceof Error && err.message.includes("LLM call failed"))) {
        await this.middleware.runError(
          syntheticCtx,
          err instanceof Error ? err : new Error(String(err))
        );
      }
      throw err;
    }
  }

  /**
   * Stream an LLM call, emitting tokens in real-time, then aggregate into
   * a full StandardChatResponse (including accumulated tool calls).
   */
  private async streamAndAggregate(
    provider: ProviderLike,
    params: import("@sschepis/llm-wrapper").StandardChatParams
  ): Promise<StandardChatResponse> {
    const stream = provider.stream({ ...params, stream: true });

    const chunks: StandardChatChunk[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);

      const delta = chunk.choices?.[0]?.delta;
      if (delta?.content) {
        this.onToken!(delta.content);
        this.bus.emit("token", { text: delta.content });
      }
    }

    async function* replay(): AsyncIterable<StandardChatChunk> {
      for (const c of chunks) yield c;
    }
    return aggregateStream(replay());
  }
}
