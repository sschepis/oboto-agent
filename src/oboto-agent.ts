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
import type { ObotoAgentConfig, AgentEventType, AgentEvent, TriageResult, ProviderLike } from "./types.js";
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
  private costTracker?: CostTracker;
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
    // CostTracker takes no constructor args; pricing is passed to getTotalCost()
    if (config.modelPricing) {
      this.costTracker = new CostTracker();
      this.modelPricing = config.modelPricing;
    }

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
    // 1. Emit user_input and record in session + context + RAG
    this.bus.emit("user_input", { text: userInput });

    const userMsg: ConversationMessage = {
      role: MessageRole.User,
      blocks: [{ kind: "text", text: userInput }],
    };
    await this.recordMessage(userMsg);
    this.bus.emit("state_updated", { reason: "user_input" });

    // 1b. Optionally augment context with RAG-retrieved past conversation
    if (this.conversationRAG) {
      try {
        const { context } = await this.conversationRAG.retrieve(userInput);
        if (context) {
          // Inject RAG context as a system-level hint
          await this.contextManager.push({
            role: "system",
            content: context,
          });
        }
      } catch (err) {
        console.warn("[ObotoAgent] RAG retrieval failed:", err instanceof Error ? err.message : err);
      }
    }

    // 2. Triage via local LLM (uses lmscript for structured output, cached)
    const triageResult = await this.triage(userInput);
    this.bus.emit("triage_result", triageResult);

    if (this.interrupted) return;

    // 3. If local can handle directly, emit and return
    if (!triageResult.escalate && triageResult.directResponse) {
      const response = triageResult.directResponse;
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

    // 4. Escalate to remote model with tool access via lmscript AgentLoop
    if (triageResult.escalate) {
      this.bus.emit("agent_thought", {
        text: triageResult.reasoning,
        model: "local",
        escalating: true,
      });
    }

    const modelName = triageResult.escalate
      ? this.config.remoteModelName
      : this.config.localModelName;

    const runtime = triageResult.escalate
      ? this.remoteRuntime
      : this.localRuntime;

    console.log("[ObotoAgent] Executing with model:", modelName, "| via lmscript AgentLoop");

    if (this.onToken) {
      // Streaming path: use raw provider streaming with token emission,
      // then use AgentLoop for the structured tool calling
      await this.executeWithStreaming(runtime, modelName, userInput);
    } else {
      // Non-streaming path: delegate entirely to lmscript AgentLoop
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

    // Define the agent task function
    // We use a permissive schema for the final output since the agent
    // produces natural text + reasoning, not a strict structured format
    const agentFn = {
      name: "agent-task",
      model: modelName,
      system: this.systemPrompt,
      prompt: (input: string) => {
        // Include conversation context in the prompt
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

    const agentConfig: AgentConfig = {
      maxIterations: this.maxIterations,
      onToolCall: (tc: ToolCall) => {
        const command = typeof tc.arguments === "object" && tc.arguments !== null
          ? (tc.arguments as Record<string, unknown>).command ?? tc.name
          : tc.name;
        const kwargs = typeof tc.arguments === "object" && tc.arguments !== null
          ? (tc.arguments as Record<string, unknown>).kwargs ?? {}
          : {};

        this.bus.emit("tool_execution_complete", {
          command,
          kwargs,
          result: typeof tc.result === "string" ? tc.result : JSON.stringify(tc.result),
        });

        // Index tool result for RAG retrieval
        const resultStr = typeof tc.result === "string" ? tc.result : JSON.stringify(tc.result);
        this.recordToolResult(String(command), kwargs as Record<string, unknown>, resultStr);
      },
      onIteration: (iteration: number, response: string) => {
        this.bus.emit("agent_thought", {
          text: response,
          model: modelName,
          iteration,
        });

        // Return false to stop early if interrupted
        if (this.interrupted) return false;
      },
    };

    const agentLoop = new AgentLoop(runtime, agentConfig);
    const result = await agentLoop.run(agentFn, userInput);

    // Record the final response
    const responseText = result.data.response;

    const assistantMsg: ConversationMessage = {
      role: MessageRole.Assistant,
      blocks: [{ kind: "text", text: responseText }],
    };
    await this.recordMessage(assistantMsg);
    this.bus.emit("state_updated", { reason: "assistant_response" });
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

    // Convert lmscript ChatMessages -> llm-wrapper Messages
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
    const turnStartTime = Date.now();
    const totalUsage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

    // ── Middleware: signal turn start ──
    // We create a synthetic ExecutionContext for middleware hooks.
    // The streaming path doesn't have an LScriptFunction, so we approximate.
    const syntheticCtx = {
      fn: { name: "streaming-turn", model: modelName, system: this.systemPrompt, prompt: () => "", schema: {} as any },
      input: _userInput,
      messages: messages as any,
      attempt: 1,
      startTime: turnStartTime,
    };
    await this.middleware.runBeforeExecute(syntheticCtx);

    try {
      for (let iteration = 1; iteration <= this.maxIterations; iteration++) {
        if (this.interrupted) break;

        // ── Budget check before each LLM call ──
        if (this.costTracker && this.budget) {
          this.costTracker.checkBudget(this.budget);
        }

        // ── Rate limit: wait for slot ──
        await this.rateLimiter?.acquire();

        const isLastIteration = iteration === this.maxIterations;

        if (isLastIteration) {
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

        // Stream and aggregate
        let response: StandardChatResponse;
        try {
          response = await this.streamAndAggregate(provider, params);
        } catch (err) {
          // ── Middleware: signal error ──
          await this.middleware.runError(
            syntheticCtx,
            err instanceof Error ? err : new Error(String(err))
          );
          console.error("[ObotoAgent] LLM call failed:", err instanceof Error ? err.message : err);
          throw err;
        }

        // ── Extract usage from the aggregated response ──
        const usage = response?.usage;
        if (usage) {
          const promptTokens = usage.prompt_tokens ?? 0;
          const completionTokens = usage.completion_tokens ?? 0;
          const usageTotal = usage.total_tokens ?? (promptTokens + completionTokens);

          totalUsage.promptTokens += promptTokens;
          totalUsage.completionTokens += completionTokens;
          totalUsage.totalTokens += usageTotal;

          // ── Rate limit: report tokens consumed ──
          this.rateLimiter?.reportTokens(usageTotal);

          // ── Unified usage tracking via bridge (records in both as-agent and lmscript trackers) ──
          this.usageBridge.recordFromLmscript(modelName, {
            promptTokens,
            completionTokens,
            totalTokens: usageTotal,
          });

          // Emit cost update event so consumers can monitor costs in real-time
          if (this.costTracker) {
            this.bus.emit("cost_update", {
              iteration,
              totalTokens: this.costTracker.getTotalTokens(),
              totalCost: this.costTracker.getTotalCost(this.modelPricing),
            });
          }
        }

        const choice = response?.choices?.[0];
        const content = (choice?.message?.content as string) ?? "";
        const toolCalls = choice?.message?.tool_calls;

        if (content) {
          this.bus.emit("agent_thought", {
            text: content,
            model: modelName,
            iteration,
          });
        }

        // If no tool calls, this is the final response
        if (!toolCalls || toolCalls.length === 0) {
          const assistantMsg: ConversationMessage = {
            role: MessageRole.Assistant,
            blocks: [{ kind: "text", text: content }],
          };
          await this.recordMessage(assistantMsg);
          this.bus.emit("state_updated", { reason: "assistant_response" });
          this.bus.emit("turn_complete", {
            model: modelName,
            escalated: true,
            iterations: iteration,
            toolCalls: totalToolCalls,
            usage: totalUsage,
          });

          // ── Middleware: signal completion ──
          await this.middleware.runComplete(syntheticCtx, {
            data: content,
            raw: content,
            usage: totalUsage,
          } as any);
          return;
        }

        // Append assistant message (with tool_calls) to conversation
        messages.push({
          role: "assistant",
          content: content || null,
          tool_calls: toolCalls,
        });

        // Execute each tool call
        for (const tc of toolCalls) {
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
              totalToolCalls++;
              continue;
            }
          }

          // Detect duplicate tool calls
          const callSig = JSON.stringify({ command, kwargs });
          const dupeCount = callHistory.filter((s) => s === callSig).length;
          callHistory.push(callSig);

          if (dupeCount >= 2) {
            messages.push({
              role: "tool",
              tool_call_id: tc.id,
              content: `You already called "${command}" with these arguments ${dupeCount} time(s). Use the data you already have.`,
            });
            totalToolCalls++;
            continue;
          }

          this.bus.emit("tool_execution_start", { command, kwargs });

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

          // ── Post-tool-use hooks ──
          if (this.hookIntegration) {
            this.hookIntegration.runPostToolUse(command, toolInputStr, truncated, isError);
          }

          this.bus.emit("tool_execution_complete", { command, kwargs, result: truncated });
          this.recordToolResult(command, kwargs, truncated);
          totalToolCalls++;

          messages.push({
            role: "tool",
            tool_call_id: tc.id,
            content: truncated,
          });
        }
      }

      // Exhausted iterations fallback
      const fallbackMsg: ConversationMessage = {
        role: MessageRole.Assistant,
        blocks: [{ kind: "text", text: "I reached the maximum number of iterations. Here is what I have so far." }],
      };
      await this.recordMessage(fallbackMsg);
      this.bus.emit("state_updated", { reason: "max_iterations" });
      this.bus.emit("turn_complete", {
        model: modelName,
        escalated: true,
        iterations: this.maxIterations,
        toolCalls: totalToolCalls,
        usage: totalUsage,
      });

      // ── Middleware: signal completion for max iterations path ──
      await this.middleware.runComplete(syntheticCtx, {
        data: "max_iterations_reached",
        raw: "",
        usage: totalUsage,
      } as any);

    } catch (err) {
      // ── Middleware: signal error if not already handled ──
      // (middleware.runError may have already been called above for LLM errors)
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
