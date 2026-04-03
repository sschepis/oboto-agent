import { z } from "zod";
import {
  LScriptRuntime,
  AgentLoop,
  type LScriptFunction,
  type ToolDefinition,
  type ChatMessage,
  type ToolCall,
} from "@sschepis/lmscript";
import type { Session, ConversationMessage } from "@sschepis/as-agent";
import { MessageRole } from "@sschepis/as-agent";
import type { ObotoAgentConfig, AgentEventType, AgentEvent, TriageResult } from "./types.js";
import { AgentEventBus } from "./event-bus.js";
import { ContextManager } from "./context-manager.js";
import { createTriageFunction, type TriageInput } from "./triage.js";
import { createRouterTool } from "./adapters/tools.js";
import { toChat, fromChat, createEmptySession } from "./adapters/memory.js";

type EventHandler = (event: AgentEvent) => void;

/** Free-form response schema for the main agent loop. */
const AgentResponseSchema = z.object({
  response: z.string().describe("The agent's response to the user"),
});

type AgentInput = { userInput: string; context: string };

/**
 * ObotoAgent is the central orchestrator for dual-LLM agent execution.
 *
 * It binds together:
 * - lmscript (LLM I/O via local and remote providers)
 * - swiss-army-tool (tool execution via Router)
 * - as-agent (session state and conversation history)
 *
 * All interaction flows through an event-driven architecture.
 */
export class ObotoAgent {
  private bus = new AgentEventBus();
  private localRuntime: LScriptRuntime;
  private remoteRuntime: LScriptRuntime;
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
    this.localRuntime = new LScriptRuntime({ provider: config.localModel });
    this.remoteRuntime = new LScriptRuntime({ provider: config.remoteModel });
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

    // 2. Triage via local LLM
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
    const runtime = triageResult.escalate ? this.remoteRuntime : this.localRuntime;
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

    await this.executeWithModel(runtime, modelName, userInput);
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

  private async executeWithModel(
    runtime: LScriptRuntime,
    modelName: string,
    userInput: string
  ): Promise<void> {
    const contextMessages = this.contextManager.getMessages();
    const contextStr = contextMessages
      .filter((m) => m.role !== "system")
      .map((m) => {
        const text = typeof m.content === "string" ? m.content : "[complex content]";
        return `${m.role}: ${text}`;
      })
      .join("\n");

    const agentFn: LScriptFunction<AgentInput, typeof AgentResponseSchema> = {
      name: "agent_execute",
      model: modelName,
      system: this.systemPrompt,
      prompt: ({ context }) =>
        context
          ? `Conversation so far:\n${context}\n\nRespond to the user's latest message. Use tools when needed.`
          : `Respond to the user.`,
      schema: AgentResponseSchema,
      temperature: 0.7,
      tools: [this.routerTool],
    };

    const agent = new AgentLoop(runtime, {
      maxIterations: this.maxIterations,
      onToolCall: (toolCall: ToolCall) => {
        const args = toolCall.arguments as { command?: string; kwargs?: Record<string, unknown> } | undefined;
        this.bus.emit("tool_execution_start", {
          command: args?.command ?? toolCall.name,
          kwargs: args?.kwargs ?? {},
        });
        this.bus.emit("tool_execution_complete", {
          command: args?.command ?? toolCall.name,
          kwargs: args?.kwargs ?? {},
          result: toolCall.result,
        });

        // Check for interruption
        if (this.interrupted) return false;
      },
      onIteration: (_iteration: number, response: string) => {
        this.bus.emit("agent_thought", {
          text: response,
          model: modelName,
          iteration: _iteration,
        });
        if (this.interrupted) return false;
      },
    });

    const result = await agent.run(agentFn, {
      userInput,
      context: contextStr,
    });

    // Record final response in session and context
    const responseText = result.data.response;
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
      iterations: result.iterations,
      toolCalls: result.toolCalls.length,
    });
  }
}
