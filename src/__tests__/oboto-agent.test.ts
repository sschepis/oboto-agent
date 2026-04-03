import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  BranchNode,
  LeafNode,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";
import { MockProvider } from "@sschepis/lmscript";
import { ObotoAgent } from "../oboto-agent.js";
import type { AgentEvent } from "../types.js";

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

  const localProvider = new MockProvider({
    defaultResponse: overrides?.localResponse ?? triageResponse,
  });

  const remoteResponse = JSON.stringify({
    response: "This is the remote model response.",
  });
  const remoteProvider = new MockProvider({
    defaultResponse: overrides?.remoteResponse ?? remoteResponse,
  });

  const agent = new ObotoAgent({
    localModel: localProvider,
    remoteModel: remoteProvider,
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
        localModel: new MockProvider(),
        remoteModel: new MockProvider(),
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

      const thoughts: AgentEvent[] = [];
      agent.on("agent_thought", (e) => thoughts.push(e));

      await agent.submitInput("Analyze the codebase");

      expect(remoteProvider.getRequestCount()).toBeGreaterThan(0);
      // Should emit an escalation thought
      const escalationThought = thoughts.find(
        (t) => (t.payload as any).escalating === true
      );
      expect(escalationThought).toBeDefined();
    });
  });

  describe("interruption", () => {
    it("emits interruption event", () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("interruption", (e) => events.push(e));

      agent.interrupt("New directive");

      expect(events).toHaveLength(1);
      expect((events[0].payload as any).newDirectives).toBe("New directive");
    });

    it("records interruption in session", () => {
      const { agent } = createTestAgent();
      agent.interrupt("Stop and do this instead");

      const session = agent.getSession();
      expect(session.messages).toHaveLength(1);
      expect(session.messages[0].blocks[0]).toEqual({
        kind: "text",
        text: "[INTERRUPTION] Stop and do this instead",
      });
    });

    it("emits state_updated on interruption", () => {
      const { agent } = createTestAgent();
      const events: AgentEvent[] = [];
      agent.on("state_updated", (e) => events.push(e));

      agent.interrupt("Change course");

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
});
