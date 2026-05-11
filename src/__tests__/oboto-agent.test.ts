import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  BranchNode,
  LeafNode,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";
import { ObotoAgent } from "../oboto-agent.js";
import type { AgentEvent } from "../types.js";
import type {
  BaseProvider,
  StandardChatParams,
  StandardChatResponse,
  StandardChatChunk,
} from "@sschepis/llm-wrapper";
import {
  ScriptedMockProvider,
  SimpleMockProvider,
  MockScript,
  ToolCallSpec,
  createStreamingTestAgent,
  createNonStreamingTestAgent,
  createTestRouter,
  collectEvents,
  TRIAGE_ESCALATE,
  TRIAGE_DIRECT,
} from "./helpers.js";

/**
 * Minimal mock that conforms to llm-wrapper's BaseProvider contract.
 * Returns OpenAI-compatible chat completion responses.
 */
class MockLLMProvider {
  readonly providerName = "mock";
  private defaultContent: string;
  private requests: StandardChatParams[] = [];

  constructor(defaultContent = "{}") {
    this.defaultContent = defaultContent;
  }

  async chat(params: StandardChatParams): Promise<StandardChatResponse> {
    this.requests.push(params);
    return {
      id: `mock-${Date.now()}`,
      object: "chat.completion" as const,
      created: Math.floor(Date.now() / 1000),
      model: params.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant" as const,
            content: this.defaultContent,
          },
          finish_reason: "stop" as const,
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
      },
    };
  }

  async *stream(
    params: StandardChatParams
  ): AsyncIterable<StandardChatChunk> {
    const response = await this.chat(params);
    yield {
      id: response.id,
      object: "chat.completion.chunk" as const,
      created: response.created,
      model: response.model,
      choices: [
        {
          index: 0,
          delta: {
            role: "assistant" as const,
            content: response.choices[0].message.content as string,
          },
          finish_reason: "stop" as const,
        },
      ],
    };
  }

  getRequestCount(): number {
    return this.requests.length;
  }

  getRequests(): StandardChatParams[] {
    return this.requests;
  }
}

function createTestAgent(overrides?: {
  localResponse?: string;
  remoteResponse?: string;
}) {
  // Build a simple router
  const root = new BranchNode({ name: "root", description: "Root" });
  root.addChild(
    new LeafNode({
      name: "echo",
      description: "Echo input",
      optionalArgs: { text: { type: "string" } },
      handler: (kwargs) => `Echo: ${kwargs.text ?? "empty"}`,
    })
  );
  const session = new SessionManager("test");
  const router = new Router(root, session);

  // Default mock responses: triage says no-escalate with direct response
  const triageResponse = JSON.stringify({
    escalate: false,
    reasoning: "Simple query",
    directResponse: "Hello! How can I help?",
  });

  const localProvider = new MockLLMProvider(
    overrides?.localResponse ?? triageResponse
  );

  const remoteProvider = new MockLLMProvider(
    overrides?.remoteResponse ?? "This is the remote model response."
  );

  const agent = new ObotoAgent({
    localModel: localProvider as unknown as BaseProvider,
    remoteModel: remoteProvider as unknown as BaseProvider,
    localModelName: "test-local",
    remoteModelName: "test-remote",
    router,
    maxContextTokens: 4096,
    maxIterations: 3,
  });

  return { agent, localProvider, remoteProvider, router };
}

describe("ObotoAgent", () => {
  describe("construction", () => {
    it("creates an agent with default config", () => {
      const { agent } = createTestAgent();
      expect(agent.processing).toBe(false);
      expect(agent.getSession().messages).toEqual([]);
    });

    it("uses provided session", () => {
      const root = new BranchNode({ name: "root", description: "Root" });
      const router = new Router(root, new SessionManager("test"));
      const existingSession = {
        version: 1,
        messages: [
          {
            role: 1, // MessageRole.User
            blocks: [{ kind: "text" as const, text: "Previous message" }],
          },
        ],
      };

      const agent = new ObotoAgent({
        localModel: new MockLLMProvider() as unknown as BaseProvider,
        remoteModel: new MockLLMProvider() as unknown as BaseProvider,
        localModelName: "test",
        remoteModelName: "test",
        router,
        session: existingSession,
      });

      expect(agent.getSession().messages).toHaveLength(1);
    });
  });

  describe("event system", () => {
    it("emits user_input on submitInput", async () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("user_input", (e) => events.push(e));

      await agent.submitInput("Hello");

      expect(events).toHaveLength(1);
      expect(events[0].payload).toEqual({ text: "Hello" });
    });

    it("emits triage_result", async () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("triage_result", (e) => events.push(e));

      await agent.submitInput("Hello");

      expect(events).toHaveLength(1);
      expect((events[0].payload as any).escalate).toBe(false);
    });

    it("emits agent_thought for direct responses", async () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("agent_thought", (e) => events.push(e));

      await agent.submitInput("Hello");

      expect(events.length).toBeGreaterThanOrEqual(1);
      expect((events[0].payload as any).text).toBe("Hello! How can I help?");
      expect((events[0].payload as any).model).toBe("local");
    });

    it("emits turn_complete", async () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("turn_complete", (e) => events.push(e));

      await agent.submitInput("Hello");

      expect(events).toHaveLength(1);
      expect((events[0].payload as any).escalated).toBe(false);
    });

    it("emits state_updated events", async () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("state_updated", (e) => events.push(e));

      await agent.submitInput("Hello");

      const reasons = events.map((e) => (e.payload as any).reason);
      expect(reasons).toContain("user_input");
      expect(reasons).toContain("assistant_response");
    });

    it("unsubscribe function works", async () => {
      const { agent } = createTestAgent();
      const handler = vi.fn();
      const unsub = agent.on("user_input", handler);

      unsub();
      await agent.submitInput("Hello");

      expect(handler).not.toHaveBeenCalled();
    });

    it("once() fires only once across multiple submits", async () => {
      const { agent } = createTestAgent();
      const handler = vi.fn();
      agent.once("user_input", handler);

      await agent.submitInput("First");
      await agent.submitInput("Second");

      expect(handler).toHaveBeenCalledOnce();
    });

    it("removeAllListeners() prevents further events", async () => {
      const { agent } = createTestAgent();
      const handler = vi.fn();
      agent.on("user_input", handler);
      agent.removeAllListeners();

      await agent.submitInput("Hello");

      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe("session management", () => {
    it("records user messages in session", async () => {
      const { agent } = createTestAgent();
      await agent.submitInput("Test message");

      const session = agent.getSession();
      expect(session.messages.length).toBeGreaterThanOrEqual(1);
      const userMsg = session.messages.find((m) => m.role === 1);
      expect(userMsg).toBeDefined();
      expect(userMsg!.blocks[0]).toEqual({ kind: "text", text: "Test message" });
    });

    it("records assistant responses in session", async () => {
      const { agent } = createTestAgent();
      await agent.submitInput("Hello");

      const session = agent.getSession();
      const assistantMsg = session.messages.find((m) => m.role === 2);
      expect(assistantMsg).toBeDefined();
    });
  });

  describe("triage and escalation", () => {
    it("handles direct response (no escalation)", async () => {
      const { agent, remoteProvider } = createTestAgent();
      await agent.submitInput("Hi there");

      // Remote should not have been called
      expect(remoteProvider.getRequestCount()).toBe(0);
    });

    it("escalates to remote model when triage says so", async () => {
      const triageResponse = JSON.stringify({
        escalate: true,
        reasoning: "Needs complex analysis",
      });
      const { agent, remoteProvider } = createTestAgent({
        localResponse: triageResponse,
      });

      const triageEvents: AgentEvent[] = [];
      agent.on("triage_result", (e) => triageEvents.push(e));
      const phaseEvents: AgentEvent[] = [];
      agent.on("phase", (e) => phaseEvents.push(e));

      await agent.submitInput("Analyze the codebase");

      expect(remoteProvider.getRequestCount()).toBeGreaterThan(0);
      // Should emit a triage_result with escalate: true
      const triageEvent = triageEvents.find(
        (t) => (t.payload as any).escalate === true
      );
      expect(triageEvent).toBeDefined();
      // Should emit a thinking phase transition
      const thinkingPhase = phaseEvents.find(
        (t) => (t.payload as any).phase === "thinking"
      );
      expect(thinkingPhase).toBeDefined();
    });
  });

  describe("interruption", () => {
    it("emits interruption event", async () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("interruption", (e) => events.push(e));

      await agent.interrupt("New directive");

      expect(events).toHaveLength(1);
      expect((events[0].payload as any).newDirectives).toBe("New directive");
    });

    it("records interruption in session", async () => {
      const { agent } = createTestAgent();
      await agent.interrupt("Stop and do this instead");

      const session = agent.getSession();
      expect(session.messages).toHaveLength(1);
      expect(session.messages[0].blocks[0]).toEqual({
        kind: "text",
        text: "[INTERRUPTION] Stop and do this instead",
      });
    });

    it("emits state_updated on interruption", async () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("state_updated", (e) => events.push(e));

      await agent.interrupt("Change course");

      expect(events).toHaveLength(1);
      expect((events[0].payload as any).reason).toBe("interruption");
    });
  });

  describe("error handling", () => {
    it("emits error event on LLM failure", async () => {
      const { agent } = createTestAgent({
        localResponse: "invalid json that will fail parsing",
      });
      const errors: AgentEvent[] = [];
      agent.on("error", (e) => errors.push(e));

      await agent.submitInput("Hello");

      expect(errors).toHaveLength(1);
      expect((errors[0].payload as any).message).toBeTypeOf("string");
    });

    it("resets isProcessing after error", async () => {
      const { agent } = createTestAgent({
        localResponse: "not json",
      });
      agent.on("error", () => {}); // prevent unhandled

      await agent.submitInput("Hello");

      expect(agent.processing).toBe(false);
    });
  });

  // ── Bug regressions ────────────────────────────────────────────────

  describe("bug regressions", () => {
    it("emits turn_complete when interrupted during triage", async () => {
      const { agent } = createStreamingTestAgent([
        { type: "text", content: "This should not be reached" },
      ]);
      const ev = collectEvents(agent, "triage_result", "turn_complete");

      // Listen for triage_result and interrupt immediately
      agent.on("triage_result", () => {
        agent.interrupt();
      });

      await agent.submitInput("Hello");

      const turnCompletes = ev.ofType("turn_complete");
      expect(turnCompletes.length).toBeGreaterThanOrEqual(1);
      const payload = turnCompletes[0].payload as any;
      expect(payload.interrupted).toBe(true);
    });

    it("emits turn_complete when LLM call fails", async () => {
      const { agent } = createStreamingTestAgent([
        { type: "error", error: new Error("LLM failed") },
      ]);
      const ev = collectEvents(agent, "error", "turn_complete");

      await agent.submitInput("Hello");

      const errors = ev.ofType("error");
      expect(errors.length).toBeGreaterThanOrEqual(1);
      expect((errors[0].payload as any).message).toContain("LLM failed");

      const turnCompletes = ev.ofType("turn_complete");
      expect(turnCompletes.length).toBeGreaterThanOrEqual(1);
    });

    it("emits turn_complete when LLM call fails (agent loop path)", async () => {
      const localProvider = new SimpleMockProvider(TRIAGE_ESCALATE);
      const remoteProvider = new ScriptedMockProvider([
        { type: "error", error: new Error("LLM failed") },
      ]);
      const router = createTestRouter();

      // Create agent WITHOUT onToken so it takes the agent loop path
      const agent = new ObotoAgent({
        localModel: localProvider as unknown as BaseProvider,
        remoteModel: remoteProvider as unknown as BaseProvider,
        localModelName: "test-local",
        remoteModelName: "test-remote",
        router,
        maxIterations: 3,
      });

      const ev = collectEvents(agent, "error", "turn_complete");

      await agent.submitInput("Hello");

      // The error may be wrapped by lmscript, so just check that an error was emitted
      const errors = ev.ofType("error");
      expect(errors.length).toBeGreaterThanOrEqual(1);
      expect((errors[0].payload as any).message).toBeTypeOf("string");

      const turnCompletes = ev.ofType("turn_complete");
      expect(turnCompletes.length).toBeGreaterThanOrEqual(1);
    });

    it("handles finish_reason=length by requesting continuation", async () => {
      const { agent } = createStreamingTestAgent([
        { type: "text", content: "Partial response that was cut", finishReason: "length" },
        { type: "text", content: "Continued and completed response." },
      ]);
      const ev = collectEvents(agent, "agent_thought");

      await agent.submitInput("Write a long essay");

      const thoughts = ev.payloads<{ text: string }>("agent_thought");
      // The first truncated response should emit a system thought about continuation
      const continuationThought = thoughts.find(
        (t) => t.text && t.text.includes("truncated")
      );
      expect(continuationThought).toBeDefined();

      // The second response should also appear
      const completedThought = thoughts.find(
        (t) => t.text && t.text.includes("Continued and completed")
      );
      expect(completedThought).toBeDefined();
    });
  });

  // ── Concurrent submission ──────────────────────────────────────────

  describe("concurrent submission", () => {
    it("submitInput while processing triggers interrupt", async () => {
      // Use a slow-responding remote so the first submitInput is still
      // processing when the second arrives.
      const { agent } = createStreamingTestAgent([
        { type: "text", content: "First response (slow)" },
        { type: "text", content: "Second response" },
      ]);
      const ev = collectEvents(agent, "interruption", "turn_complete");

      // Intercept triage_result to inject a second submitInput while processing
      agent.on("triage_result", () => {
        // At this point isProcessing is true. Calling submitInput
        // synchronously triggers the interrupt path.
        agent.submitInput("Second input");
      });

      await agent.submitInput("First input");

      const interruptions = ev.ofType("interruption");
      expect(interruptions.length).toBeGreaterThanOrEqual(1);
      expect((interruptions[0].payload as any).newDirectives).toBe("Second input");
    });

    it("processing flag prevents concurrent execution", async () => {
      const { agent } = createStreamingTestAgent([
        { type: "text", content: "Response" },
      ]);

      let capturedProcessing = false;
      const ev = collectEvents(agent, "interruption");

      agent.on("triage_result", () => {
        // Capture the processing flag during execution
        capturedProcessing = agent.processing;
        // Second submission should go through interrupt path
        agent.submitInput("Concurrent input");
      });

      await agent.submitInput("First");

      expect(capturedProcessing).toBe(true);

      const interruptions = ev.ofType("interruption");
      expect(interruptions.length).toBeGreaterThanOrEqual(1);
    });
  });

  // ── Slash commands ─────────────────────────────────────────────────

  describe("slash commands", () => {
    it("executes custom slash command", async () => {
      const { agent } = createNonStreamingTestAgent();
      const ev = collectEvents(agent, "slash_command");

      // Register a custom slash command
      agent.getSlashCommands().registerCommand(
        {
          name: "test",
          summary: "A test command",
          argumentHint: "<text>",
          resumeSupported: false,
        },
        (args: string) => `Test result: ${args}`,
      );

      await agent.submitInput("/test hello");

      const slashEvents = ev.ofType("slash_command");
      expect(slashEvents).toHaveLength(1);
      const payload = slashEvents[0].payload as any;
      expect(payload.command).toBe("test");
      expect(payload.args).toBe("hello");
      expect(payload.result).toBe("Test result: hello");
    });

    it("falls through for unregistered slash commands", async () => {
      const { agent } = createNonStreamingTestAgent();
      const ev = collectEvents(agent, "triage_result", "slash_command");

      await agent.submitInput("/notregistered");

      // Should NOT emit a slash_command event
      const slashEvents = ev.ofType("slash_command");
      expect(slashEvents).toHaveLength(0);

      // Should fall through to triage
      const triageEvents = ev.ofType("triage_result");
      expect(triageEvents.length).toBeGreaterThanOrEqual(1);
    });
  });

  // ── Session management ─────────────────────────────────────────────

  describe("session management (extended)", () => {
    it("syncSession replaces session and context", async () => {
      const { agent } = createNonStreamingTestAgent();

      // Submit input to populate the session
      await agent.submitInput("Original message");
      const originalSession = agent.getSession();
      expect(originalSession.messages.length).toBeGreaterThan(0);

      // Create a new session with different content
      const newSession = {
        version: 1,
        messages: [
          {
            role: 1, // MessageRole.User
            blocks: [{ kind: "text" as const, text: "Replaced session message" }],
          },
        ],
      };

      await agent.syncSession(newSession);

      const synced = agent.getSession();
      expect(synced.messages).toHaveLength(1);
      expect(synced.messages[0].blocks[0]).toEqual({
        kind: "text",
        text: "Replaced session message",
      });
    });
  });

  // ── Cost tracking API ──────────────────────────────────────────────

  describe("cost tracking API", () => {
    it("getCostSummary returns token counts after a turn", async () => {
      const { agent } = createStreamingTestAgent([
        { type: "text", content: "Here is my answer." },
      ]);

      await agent.submitInput("What is 2+2?");

      const summary = agent.getCostSummary();
      expect(summary).toBeDefined();
      expect(summary!.totalTokens).toBeGreaterThan(0);
    });
  });

  // ── Public API ─────────────────────────────────────────────────────

  describe("public API", () => {
    it("setOnToken updates streaming callback", async () => {
      // Start with a non-streaming agent (no onToken)
      const localProvider = new SimpleMockProvider(TRIAGE_ESCALATE);
      const remoteProvider = new ScriptedMockProvider([
        { type: "text", content: "Streamed response" },
      ]);
      const router = createTestRouter();
      const agent = new ObotoAgent({
        localModel: localProvider as unknown as BaseProvider,
        remoteModel: remoteProvider as unknown as BaseProvider,
        localModelName: "test-local",
        remoteModelName: "test-remote",
        router,
        maxIterations: 3,
      });

      // Attach onToken dynamically via setOnToken
      const tokens: string[] = [];
      agent.setOnToken((t) => tokens.push(t));

      await agent.submitInput("Stream this");

      // Tokens should have been received via the newly set callback
      expect(tokens.length).toBeGreaterThan(0);
      expect(tokens.join("")).toContain("Streamed response");
    });

    it("compactSession returns null when no compactor configured", () => {
      const { agent } = createNonStreamingTestAgent();
      const result = agent.compactSession();
      expect(result).toBeNull();
    });
  });
});
