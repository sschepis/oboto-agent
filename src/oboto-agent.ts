import { LScriptRuntime, type ToolDefinition, type ChatMessage } from "@sschepis/lmscript";
import type {
  BaseProvider,
  StandardChatParams,
  Message as WrapperMessage,
  ToolDefinition as WrapperToolDef,
  StandardChatResponse,
} from "@sschepis/llm-wrapper";
import type { Session, ConversationMessage } from "@sschepis/as-agent";
import { MessageRole } from "@sschepis/as-agent";
import type { ObotoAgentConfig, AgentEventType, AgentEvent, TriageResult } from "./types.js";
import { AgentEventBus } from "./event-bus.js";
import { ContextManager } from "./context-manager.js";
import { createTriageFunction } from "./triage.js";
import { createRouterTool } from "./adapters/tools.js";
import { toLmscriptProvider } from "./adapters/llm-wrapper.js";
import { toChat, createEmptySession } from "./adapters/memory.js";

type EventHandler = (event: AgentEvent) => void;

/**
 * ObotoAgent is the central orchestrator for dual-LLM agent execution.
 *
 * It binds together:
 * - llm-wrapper (LLM communication via local and remote providers)
 * - lmscript (structured/schema-validated calls for triage)
 * - swiss-army-tool (tool execution via Router)
 * - as-agent (session state and conversation history)
 *
 * All interaction flows through an event-driven architecture.
 */
export class ObotoAgent {
  private bus = new AgentEventBus();
  private localRuntime: LScriptRuntime;
  private localProvider: BaseProvider;
  private remoteProvider: BaseProvider;
  private contextManager: ContextManager;
  private routerTool: ToolDefinition<any, any>;
  private triageFn: ReturnType<typeof createTriageFunction>;
  private session: Session;
  private isProcessing = false;
  private interrupted = false;
  private systemPrompt: string;
  private maxIterations: number;
  private config: ObotoAgentConfig;

  constructor(config: ObotoAgentConfig) {
    this.config = config;
    this.localProvider = config.localModel;
    this.remoteProvider = config.remoteModel;

    // Wrap llm-wrapper providers into lmscript LLMProvider for structured calls (triage)
    const localLmscript = toLmscriptProvider(config.localModel, "local");
    this.localRuntime = new LScriptRuntime({ provider: localLmscript });

    this.session = config.session ?? createEmptySession();
    this.systemPrompt = config.systemPrompt ?? "You are a helpful AI assistant with access to tools.";
    this.maxIterations = config.maxIterations ?? 10;

    this.contextManager = new ContextManager(
      this.localRuntime,
      config.localModelName,
      config.maxContextTokens ?? 8192
    );

    this.routerTool = createRouterTool(config.router);
    this.triageFn = createTriageFunction(config.localModelName);

    // Push system prompt into context
    this.contextManager.push({
      role: "system",
      content: this.systemPrompt,
    });
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

    this.isProcessing = true;
    this.interrupted = false;

    try {
      await this.executionLoop(text);
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
  interrupt(newDirectives?: string): void {
    this.interrupted = true;
    this.bus.emit("interruption", { newDirectives });

    if (newDirectives) {
      const msg: ConversationMessage = {
        role: MessageRole.User,
        blocks: [{ kind: "text", text: `[INTERRUPTION] ${newDirectives}` }],
      };
      this.session.messages.push(msg);
      this.contextManager.push(toChat(msg));
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

  /** Remove all event listeners. */
  removeAllListeners(): void {
    this.bus.removeAllListeners();
  }

  // ── Internal ───────────────────────────────────────────────────────

  private async executionLoop(userInput: string): Promise<void> {
    // 1. Emit user_input and record in session + context
    this.bus.emit("user_input", { text: userInput });

    const userMsg: ConversationMessage = {
      role: MessageRole.User,
      blocks: [{ kind: "text", text: userInput }],
    };
    this.session.messages.push(userMsg);
    await this.contextManager.push(toChat(userMsg));
    this.bus.emit("state_updated", { reason: "user_input" });

    // 2. Triage via local LLM (uses lmscript for structured output)
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
      this.session.messages.push(assistantMsg);
      await this.contextManager.push(toChat(assistantMsg));
      this.bus.emit("state_updated", { reason: "assistant_response" });
      this.bus.emit("turn_complete", { model: "local", escalated: false });
      return;
    }

    // 4. Escalate to remote model with tool access
    const provider = triageResult.escalate ? this.remoteProvider : this.localProvider;
    const modelName = triageResult.escalate
      ? this.config.remoteModelName
      : this.config.localModelName;

    if (triageResult.escalate) {
      this.bus.emit("agent_thought", {
        text: triageResult.reasoning,
        model: "local",
        escalating: true,
      });
    }

    await this.executeWithModel(provider, modelName, userInput);
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
   * Execute the agent loop using llm-wrapper directly.
   * No JSON mode, no schema enforcement — just natural chat with tool calling.
   */
  private async executeWithModel(
    provider: BaseProvider,
    modelName: string,
    _userInput: string
  ): Promise<void> {
    // Build messages from context manager (includes system prompt + history)
    const contextMessages = this.contextManager.getMessages();

    // Convert lmscript ChatMessages → llm-wrapper Messages
    const messages: WrapperMessage[] = contextMessages.map((m) => ({
      role: m.role,
      content:
        typeof m.content === "string"
          ? m.content
          : (m.content as Array<{ type: string; text?: string }>)
              .filter((b) => b.type === "text")
              .map((b) => b.text ?? "")
              .join("\n"),
    }));

    // Build llm-wrapper tool definitions from the router tool
    const tool = this.routerTool;
    const tools: WrapperToolDef[] = [
      {
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters
            ? JSON.parse(JSON.stringify(tool.parameters))
            : { type: "object", properties: {} },
        },
      },
    ];

    let totalToolCalls = 0;

    for (let iteration = 1; iteration <= this.maxIterations; iteration++) {
      if (this.interrupted) break;

      const params: StandardChatParams = {
        model: modelName,
        messages: [...messages],
        temperature: 0.7,
        tools,
        tool_choice: "auto",
      };

      const response: StandardChatResponse = await provider.chat(params);
      const choice = response.choices[0];
      const content = (choice?.message?.content as string) ?? "";

      // Emit thought for each iteration
      this.bus.emit("agent_thought", {
        text: content,
        model: modelName,
        iteration,
      });

      // If no tool calls, this is the final response
      const toolCalls = choice?.message?.tool_calls;
      if (!toolCalls || toolCalls.length === 0) {
        const responseText = content;

        const assistantMsg: ConversationMessage = {
          role: MessageRole.Assistant,
          blocks: [{ kind: "text", text: responseText }],
        };
        this.session.messages.push(assistantMsg);
        await this.contextManager.push(toChat(assistantMsg));
        this.bus.emit("state_updated", { reason: "assistant_response" });
        this.bus.emit("turn_complete", {
          model: modelName,
          escalated: true,
          iterations: iteration,
          toolCalls: totalToolCalls,
        });
        return;
      }

      // Append assistant message (with tool_calls) to conversation
      messages.push({
        role: "assistant",
        content: content || null,
        tool_calls: toolCalls,
      });

      // Execute each tool call and feed results back
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

        this.bus.emit("tool_execution_start", { command, kwargs });

        let result: string;
        try {
          result = await tool.execute(args);
        } catch (err) {
          result = `Error: ${err instanceof Error ? err.message : String(err)}`;
        }

        this.bus.emit("tool_execution_complete", { command, kwargs, result });
        totalToolCalls++;

        // Feed tool result back as a tool message (OpenAI format)
        messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: typeof result === "string" ? result : JSON.stringify(result),
        });
      }
    }

    // Exhausted iterations
    const fallbackMsg: ConversationMessage = {
      role: MessageRole.Assistant,
      blocks: [{ kind: "text", text: "I reached the maximum number of iterations. Here is what I have so far." }],
    };
    this.session.messages.push(fallbackMsg);
    await this.contextManager.push(toChat(fallbackMsg));
    this.bus.emit("state_updated", { reason: "max_iterations" });
    this.bus.emit("turn_complete", {
      model: modelName,
      escalated: true,
      iterations: this.maxIterations,
      toolCalls: totalToolCalls,
    });
  }
}
