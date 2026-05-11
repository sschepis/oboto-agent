import { vi } from "vitest";
import {
  BranchNode,
  LeafNode,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";
import { ObotoAgent } from "../oboto-agent.js";
import type { AgentEvent, AgentEventType, ObotoAgentConfig } from "../types.js";
import type {
  BaseProvider,
  StandardChatParams,
  StandardChatResponse,
  StandardChatChunk,
} from "@sschepis/llm-wrapper";

// ── Response script types ──────────────────────────────────────────

export interface ToolCallSpec {
  name: string;
  arguments: Record<string, unknown>;
}

export type MockScript =
  | { type: "text"; content: string; finishReason?: string }
  | { type: "tool_call"; content?: string; toolCalls: ToolCallSpec[]; finishReason?: string }
  | { type: "error"; error: Error }
  | { type: "hang" }
  | { type: "empty" };

// ── ScriptedMockProvider ───────────────────────────────────────────

export class ScriptedMockProvider {
  readonly providerName = "scripted-mock";
  private scripts: MockScript[];
  private callIndex = 0;
  private requests: StandardChatParams[] = [];

  constructor(scripts: MockScript[]) {
    this.scripts = [...scripts];
  }

  private nextScript(): MockScript {
    if (this.callIndex >= this.scripts.length) {
      return { type: "text", content: "No more scripts available." };
    }
    return this.scripts[this.callIndex++];
  }

  async chat(params: StandardChatParams): Promise<StandardChatResponse> {
    this.requests.push(params);
    const script = this.nextScript();

    if (script.type === "error") {
      throw script.error;
    }

    if (script.type === "hang") {
      return new Promise<never>(() => {});
    }

    const response = this.buildResponse(script, params.model);
    return response;
  }

  async *stream(params: StandardChatParams): AsyncIterable<StandardChatChunk> {
    this.requests.push(params);
    const script = this.nextScript();

    if (script.type === "error") {
      throw script.error;
    }

    if (script.type === "hang") {
      yield* this.hangForever();
      return;
    }

    const response = this.buildResponse(script, params.model);
    const choice = response.choices[0];

    if (choice.message.content) {
      yield {
        id: response.id,
        object: "chat.completion.chunk" as const,
        created: response.created,
        model: response.model,
        choices: [{
          index: 0,
          delta: { role: "assistant" as const, content: choice.message.content as string },
          finish_reason: null as any,
        }],
      };
    }

    yield {
      id: response.id,
      object: "chat.completion.chunk" as const,
      created: response.created,
      model: response.model,
      choices: [{
        index: 0,
        delta: choice.message.tool_calls
          ? { tool_calls: choice.message.tool_calls.map((tc: any, i: number) => ({ ...tc, index: i })) }
          : {},
        finish_reason: choice.finish_reason,
      }],
      usage: response.usage,
    };
  }

  getRequestCount(): number {
    return this.requests.length;
  }

  getRequests(): StandardChatParams[] {
    return this.requests;
  }

  getCallIndex(): number {
    return this.callIndex;
  }

  private async *hangForever(): AsyncIterable<StandardChatChunk> {
    yield {
      id: "hang-start",
      object: "chat.completion.chunk" as const,
      created: Math.floor(Date.now() / 1000),
      model: "hang",
      choices: [{
        index: 0,
        delta: { role: "assistant" as const, content: "partial..." },
        finish_reason: null as any,
      }],
    };
    await new Promise<never>(() => {});
  }

  private buildResponse(script: MockScript, model: string): StandardChatResponse {
    const id = `mock-${Date.now()}-${this.callIndex}`;
    const created = Math.floor(Date.now() / 1000);

    if (script.type === "empty") {
      return {
        id, object: "chat.completion" as const, created, model,
        choices: [{
          index: 0,
          message: { role: "assistant" as const, content: "" },
          finish_reason: "stop" as const,
        }],
        usage: { prompt_tokens: 5, completion_tokens: 0, total_tokens: 5 },
      };
    }

    if (script.type === "text") {
      return {
        id, object: "chat.completion" as const, created, model,
        choices: [{
          index: 0,
          message: { role: "assistant" as const, content: script.content },
          finish_reason: (script.finishReason ?? "stop") as any,
        }],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
      };
    }

    if (script.type === "tool_call") {
      const toolCalls = script.toolCalls.map((tc, i) => ({
        id: `call-${this.callIndex}-${i}`,
        type: "function" as const,
        function: {
          name: "router_tool",
          arguments: JSON.stringify({ command: tc.name, kwargs: tc.arguments }),
        },
      }));

      return {
        id, object: "chat.completion" as const, created, model,
        choices: [{
          index: 0,
          message: {
            role: "assistant" as const,
            content: script.content ?? null,
            tool_calls: toolCalls,
          },
          finish_reason: (script.finishReason ?? "tool_calls") as any,
        }],
        usage: { prompt_tokens: 10, completion_tokens: 30, total_tokens: 40 },
      };
    }

    return {
      id, object: "chat.completion" as const, created, model,
      choices: [{
        index: 0,
        message: { role: "assistant" as const, content: "" },
        finish_reason: "stop" as const,
      }],
      usage: { prompt_tokens: 5, completion_tokens: 0, total_tokens: 5 },
    };
  }
}

// ── SimpleMockProvider (for triage — always chat, never stream) ────

export class SimpleMockProvider {
  readonly providerName = "simple-mock";
  private content: string;

  constructor(content: string) {
    this.content = content;
  }

  async chat(params: StandardChatParams): Promise<StandardChatResponse> {
    return {
      id: `simple-${Date.now()}`,
      object: "chat.completion" as const,
      created: Math.floor(Date.now() / 1000),
      model: params.model,
      choices: [{
        index: 0,
        message: { role: "assistant" as const, content: this.content },
        finish_reason: "stop" as const,
      }],
      usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    };
  }

  async *stream(params: StandardChatParams): AsyncIterable<StandardChatChunk> {
    const response = await this.chat(params);
    yield {
      id: response.id,
      object: "chat.completion.chunk" as const,
      created: response.created,
      model: response.model,
      choices: [{
        index: 0,
        delta: { role: "assistant" as const, content: this.content },
        finish_reason: "stop" as const,
      }],
      usage: response.usage,
    };
  }
}

// ── Router factories ──────────────────────────────────────────────

export function createTestRouter() {
  const root = new BranchNode({ name: "root", description: "Root" });

  root.addChild(
    new LeafNode({
      name: "echo",
      description: "Echo input back",
      optionalArgs: { text: { type: "string" } },
      handler: (kwargs) => `Echo: ${kwargs.text ?? "empty"}`,
    })
  );

  root.addChild(
    new LeafNode({
      name: "failing_tool",
      description: "Always fails",
      optionalArgs: { reason: { type: "string" } },
      handler: (kwargs) => {
        throw new Error(kwargs.reason as string ?? "Tool failed");
      },
    })
  );

  root.addChild(
    new LeafNode({
      name: "big_output",
      description: "Returns a large string",
      optionalArgs: { size: { type: "number" } },
      handler: (kwargs) => "X".repeat(Number(kwargs.size) ?? 100),
    })
  );

  const session = new SessionManager("test");
  return new Router(root, session);
}

// ── Agent factories ───────────────────────────────────────────────

const TRIAGE_ESCALATE = JSON.stringify({
  escalate: true,
  reasoning: "Needs tools",
});

const TRIAGE_DIRECT = JSON.stringify({
  escalate: false,
  reasoning: "Simple query",
  directResponse: "Direct answer from triage.",
});

export function createStreamingTestAgent(
  remoteScripts: MockScript[],
  opts?: Partial<ObotoAgentConfig> & { triageResponse?: string }
) {
  const router = opts?.router ?? createTestRouter();
  const tokens: string[] = [];
  const triageContent = opts?.triageResponse ?? TRIAGE_ESCALATE;

  const localProvider = new SimpleMockProvider(triageContent);
  const remoteProvider = new ScriptedMockProvider(remoteScripts);

  const agent = new ObotoAgent({
    localModel: localProvider as unknown as BaseProvider,
    remoteModel: remoteProvider as unknown as BaseProvider,
    localModelName: "test-local",
    remoteModelName: "test-remote",
    router,
    maxIterations: opts?.maxIterations ?? 5,
    maxOutputTokens: opts?.maxOutputTokens ?? 4096,
    onToken: (t) => tokens.push(t),
    ...opts,
  });

  return { agent, localProvider, remoteProvider, router, tokens };
}

export function createNonStreamingTestAgent(
  opts?: Partial<ObotoAgentConfig> & {
    localResponse?: string;
    remoteResponse?: string;
  }
) {
  const router = opts?.router ?? createTestRouter();

  const triageResponse = opts?.localResponse ?? TRIAGE_DIRECT;
  const remoteContent = opts?.remoteResponse ?? "Remote model response.";

  const localProvider = new SimpleMockProvider(triageResponse);
  const remoteProvider = new SimpleMockProvider(remoteContent);

  const agent = new ObotoAgent({
    localModel: localProvider as unknown as BaseProvider,
    remoteModel: remoteProvider as unknown as BaseProvider,
    localModelName: "test-local",
    remoteModelName: "test-remote",
    router,
    maxIterations: opts?.maxIterations ?? 3,
    ...opts,
  });

  return { agent, localProvider, remoteProvider, router };
}

// ── Event collector ───────────────────────────────────────────────

export function collectEvents(agent: ObotoAgent, ...types: AgentEventType[]) {
  const events: AgentEvent[] = [];
  const targetTypes = new Set(types);

  const ALL_EVENT_TYPES: AgentEventType[] = [
    "user_input", "agent_thought", "token", "phase", "triage_result",
    "tool_execution_start", "tool_execution_complete", "tool_round_complete",
    "state_updated", "interruption", "error", "cost_update", "turn_complete",
    "permission_denied", "session_compacted", "hook_denied", "hook_message",
    "router_event", "slash_command", "doom_loop",
  ];

  const subscribeTypes = targetTypes.size > 0 ? [...targetTypes] : ALL_EVENT_TYPES;

  for (const type of subscribeTypes) {
    agent.on(type, (e) => events.push(e));
  }

  return {
    all: events,
    ofType: (t: AgentEventType) => events.filter((e) => e.type === t),
    payloads: <T = unknown>(t: AgentEventType) =>
      events.filter((e) => e.type === t).map((e) => e.payload as T),
  };
}

export { TRIAGE_ESCALATE, TRIAGE_DIRECT };
