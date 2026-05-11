import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  ScriptedMockProvider,
  SimpleMockProvider,
  createStreamingTestAgent,
  createTestRouter,
  collectEvents,
  TRIAGE_ESCALATE,
} from "./helpers.js";
import type { MockScript, ToolCallSpec } from "./helpers.js";
import type { AgentEvent, DoomLoopEvent } from "../types.js";

// ── Shorthand script builders ────────────────────────────────────────

function text(content: string, finishReason?: string): MockScript {
  return { type: "text", content, finishReason };
}

function toolCall(
  name: string,
  args: Record<string, unknown>,
  content?: string,
  finishReason?: string,
): MockScript {
  return {
    type: "tool_call",
    content,
    toolCalls: [{ name, arguments: args }],
    finishReason,
  };
}

function multiToolCall(
  calls: ToolCallSpec[],
  content?: string,
): MockScript {
  return { type: "tool_call", content, toolCalls: calls };
}

function empty(): MockScript {
  return { type: "empty" };
}

function hang(): MockScript {
  return { type: "hang" };
}

// ── Tests ────────────────────────────────────────────────────────────

describe("executeWithStreaming", () => {
  // ── basic streaming flow ──────────────────────────────────────────

  describe("basic streaming flow", () => {
    it("streams tokens to onToken callback", async () => {
      const { agent, tokens } = createStreamingTestAgent([
        text("Hello"),
      ]);

      await agent.submitInput("test input");

      expect(tokens).toContain("Hello");
    });

    it("emits turn_complete with usage info", async () => {
      const { agent } = createStreamingTestAgent([
        text("Hello"),
      ]);
      const collector = collectEvents(agent, "turn_complete");

      await agent.submitInput("test input");

      const payloads = collector.payloads<Record<string, unknown>>("turn_complete");
      expect(payloads).toHaveLength(1);
      expect(payloads[0]).toHaveProperty("iterations");
      expect(payloads[0]).toHaveProperty("toolCalls");
    });

    it("records assistant response in session", async () => {
      const { agent } = createStreamingTestAgent([
        text("Hello from remote"),
      ]);

      await agent.submitInput("test input");

      const session = agent.getSession();
      // MessageRole.Assistant = 2
      const assistantMsgs = session.messages.filter((m) => m.role === 2);
      expect(assistantMsgs.length).toBeGreaterThanOrEqual(1);
      const hasContent = assistantMsgs.some((m) =>
        m.blocks.some(
          (b: any) => b.kind === "text" && b.text === "Hello from remote",
        ),
      );
      expect(hasContent).toBe(true);
    });

    it("emits agent_thought for AI text content", async () => {
      const { agent } = createStreamingTestAgent([
        text("Thinking out loud"),
      ]);
      const collector = collectEvents(agent, "agent_thought");

      await agent.submitInput("test input");

      const thoughts = collector.payloads<{ text: string; model: string }>(
        "agent_thought",
      );
      const remoteTh = thoughts.find(
        (t) => t.text === "Thinking out loud" && t.model === "test-remote",
      );
      expect(remoteTh).toBeDefined();
    });
  });

  // ── tool execution ────────────────────────────────────────────────

  describe("tool execution", () => {
    it("executes tool calls from LLM response", async () => {
      const { agent } = createStreamingTestAgent([
        toolCall("echo", { text: "hi" }),
        text("done"),
      ]);
      const collector = collectEvents(
        agent,
        "tool_execution_start",
        "tool_execution_complete",
      );

      await agent.submitInput("test input");

      const starts = collector.ofType("tool_execution_start");
      const completes = collector.ofType("tool_execution_complete");
      expect(starts.length).toBeGreaterThanOrEqual(1);
      expect(completes.length).toBeGreaterThanOrEqual(1);

      const completePayload = completes[0].payload as any;
      expect(completePayload.result).toContain("Echo: hi");
    });

    it("handles multiple tool calls in single response", async () => {
      const { agent } = createStreamingTestAgent([
        multiToolCall([
          { name: "echo", arguments: { text: "first" } },
          { name: "echo", arguments: { text: "second" } },
        ]),
        text("done"),
      ]);
      const collector = collectEvents(agent, "tool_execution_complete");

      await agent.submitInput("test input");

      const completes = collector.ofType("tool_execution_complete");
      expect(completes.length).toBeGreaterThanOrEqual(2);
      const results = completes.map((e) => (e.payload as any).result);
      expect(results.some((r: string) => r.includes("Echo: first"))).toBe(true);
      expect(results.some((r: string) => r.includes("Echo: second"))).toBe(true);
    });

    it("handles tool execution errors", async () => {
      const { agent } = createStreamingTestAgent([
        toolCall("failing_tool", {}),
        text("recovered"),
      ]);
      const collector = collectEvents(agent, "tool_execution_complete");

      await agent.submitInput("test input");

      const completes = collector.ofType("tool_execution_complete");
      expect(completes.length).toBeGreaterThanOrEqual(1);
      // The Router catches handler errors and returns a formatted error
      // string (e.g. "[SystemFault] Tool failed") rather than throwing,
      // so the result contains the error text.
      const errorComplete = completes.find(
        (e) => {
          const p = e.payload as any;
          return (p.result && typeof p.result === "string" && p.result.includes("Tool failed")) ||
                 (p.error != null);
        },
      );
      expect(errorComplete).toBeDefined();

      // Agent should continue and produce a final response
      const session = agent.getSession();
      const assistantMsgs = session.messages.filter((m) => m.role === 2);
      expect(assistantMsgs.length).toBeGreaterThanOrEqual(1);
    });

    it("truncates long tool results", async () => {
      const { agent } = createStreamingTestAgent([
        toolCall("big_output", { size: 10000 }),
        text("done"),
      ]);
      const collector = collectEvents(agent, "tool_execution_complete");

      await agent.submitInput("test input");

      const completes = collector.ofType("tool_execution_complete");
      expect(completes.length).toBeGreaterThanOrEqual(1);
      const result = (completes[0].payload as any).result as string;
      expect(result).toContain("truncated");
    });

    it("persists tool interactions in session", async () => {
      const { agent } = createStreamingTestAgent([
        toolCall("echo", { text: "persisted" }),
        text("done"),
      ]);

      await agent.submitInput("test input");

      const session = agent.getSession();
      const allBlocks = session.messages.flatMap((m) => m.blocks);
      const toolUseBlock = allBlocks.find((b: any) => b.kind === "tool_use");
      const toolResultBlock = allBlocks.find(
        (b: any) => b.kind === "tool_result",
      );
      expect(toolUseBlock).toBeDefined();
      expect(toolResultBlock).toBeDefined();
    });
  });

  // ── doom loop detection ───────────────────────────────────────────

  describe("doom loop detection", () => {
    it("injects dedup message on 2 duplicates", async () => {
      // 3 identical tool calls to trigger the dedup message (dupeCount >= 2 on 3rd call)
      const { agent, remoteProvider } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          text("final answer"),
          text("fallback"),
        ],
        { maxIterations: 10 },
      );

      await agent.submitInput("test input");

      // Check the requests sent to the provider to see if dedup messages were injected
      const requests = remoteProvider.getRequests();
      const allMessages = requests.flatMap((r) => r.messages);
      const dedupMsg = allMessages.find(
        (m: any) =>
          m.role === "tool" &&
          typeof m.content === "string" &&
          m.content.includes("You already called"),
      );
      expect(dedupMsg).toBeDefined();
    });

    it("triggers doom loop redirect at consecutive duplicates", async () => {
      // The consecutive counter resets when dupeCount < 2 (calls 1-2), so it
      // only starts accumulating from call 3 onward. The redirect fires when
      // prevConsec >= 3, which requires 6 identical calls total:
      //   call 1-2: dupeCount<2 -> consecutiveDupes reset to 0
      //   call 3:   dupeCount=2, prevConsec=0 -> dedup
      //   call 4:   dupeCount=3, prevConsec=1 -> dedup
      //   call 5:   dupeCount=4, prevConsec=2 -> dedup
      //   call 6:   dupeCount=5, prevConsec=3 -> redirect!
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          text("final"),
          text("fallback"),
        ],
        { maxIterations: 10 },
      );
      const collector = collectEvents(agent, "doom_loop");

      await agent.submitInput("test input");

      const doomEvents = collector.payloads<DoomLoopEvent>("doom_loop");
      const redirected = doomEvents.find((d) => d.redirected === true);
      expect(redirected).toBeDefined();
    });

    it("terminates on persistent doom loop at many consecutive duplicates", async () => {
      // After the redirect fires on call 6 (prevConsec=3), the persistent
      // termination fires when prevConsec >= 5, which needs call 8:
      //   call 7: prevConsec=4 -> dedup
      //   call 8: prevConsec=5 -> terminate!
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          toolCall("echo", { text: "same" }),
          text("should not reach"),
        ],
        { maxIterations: 12 },
      );
      const collector = collectEvents(agent, "doom_loop", "turn_complete");

      await agent.submitInput("test input");

      const doomEvents = collector.payloads<DoomLoopEvent>("doom_loop");
      const terminated = doomEvents.find((d) => d.redirected === false);
      expect(terminated).toBeDefined();

      const turnCompletes = collector.ofType("turn_complete");
      expect(turnCompletes.length).toBeGreaterThanOrEqual(1);
    });

    it("resets counter on novel call", async () => {
      // Pattern: echo(a), echo(a), echo(b), echo(a), then text
      // The echo(b) in between should reset the consecutive counter
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "a" }),
          toolCall("echo", { text: "a" }),
          toolCall("echo", { text: "b" }),
          toolCall("echo", { text: "a" }),
          text("done"),
          text("fallback"),
        ],
        { maxIterations: 10 },
      );
      const collector = collectEvents(agent, "doom_loop");

      await agent.submitInput("test input");

      const doomEvents = collector.ofType("doom_loop");
      expect(doomEvents).toHaveLength(0);
    });
  });

  // ── adaptive continuation ─────────────────────────────────────────

  describe("adaptive continuation", () => {
    it("extends iterations when shouldContinue returns true", async () => {
      let continueCallCount = 0;
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }),
          toolCall("echo", { text: "2" }),
          toolCall("echo", { text: "3" }),
          text("final answer"),
          text("fallback"),
        ],
        {
          maxIterations: 2,
          maxTotalIterations: 10,
          shouldContinue: async () => {
            continueCallCount++;
            // Grant extension on first call only
            return continueCallCount <= 1;
          },
        },
      );
      const collector = collectEvents(agent, "turn_complete");

      await agent.submitInput("test input");

      const payloads = collector.payloads<{ iterations: number }>(
        "turn_complete",
      );
      expect(payloads).toHaveLength(1);
      expect(payloads[0].iterations).toBeGreaterThan(2);
    });

    it("stops when shouldContinue returns false", async () => {
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }),
          toolCall("echo", { text: "2" }),
          toolCall("echo", { text: "3" }),
          text("should not reach"),
          text("fallback"),
        ],
        {
          maxIterations: 2,
          maxTotalIterations: 10,
          shouldContinue: async () => false,
        },
      );
      const collector = collectEvents(agent, "turn_complete");

      await agent.submitInput("test input");

      const payloads = collector.payloads<{ iterations: number }>(
        "turn_complete",
      );
      expect(payloads).toHaveLength(1);
      expect(payloads[0].iterations).toBeLessThanOrEqual(2);
    });

    it("handles shouldContinue callback errors gracefully", async () => {
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }),
          toolCall("echo", { text: "2" }),
          text("fallback"),
          text("extra fallback"),
        ],
        {
          maxIterations: 2,
          maxTotalIterations: 10,
          shouldContinue: async () => {
            throw new Error("callback exploded");
          },
        },
      );
      const collector = collectEvents(agent, "turn_complete", "error");

      await agent.submitInput("test input");

      // Should not crash — turn_complete should still fire
      const turnCompletes = collector.ofType("turn_complete");
      expect(turnCompletes.length).toBeGreaterThanOrEqual(1);

      // No unhandled error event (the shouldContinue error is swallowed)
      const errors = collector.ofType("error");
      expect(errors).toHaveLength(0);
    });
  });

  // ── max iterations fallback ───────────────────────────────────────

  describe("max iterations fallback", () => {
    it("emits fallback when all iterations used", async () => {
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }),
          toolCall("echo", { text: "2" }),
          // No text response — iterations exhausted
        ],
        { maxIterations: 2 },
      );
      const collector = collectEvents(agent, "turn_complete");

      await agent.submitInput("test input");

      const payloads = collector.ofType("turn_complete");
      expect(payloads).toHaveLength(1);
    });

    it("uses last assistant content as fallback", async () => {
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }, "partial thinking"),
          toolCall("echo", { text: "2" }, "more thinking"),
        ],
        { maxIterations: 2 },
      );

      await agent.submitInput("test input");

      const session = agent.getSession();
      const assistantMsgs = session.messages.filter((m) => m.role === 2);
      // The last assistant message should contain the last content
      const lastAssistant = assistantMsgs[assistantMsgs.length - 1];
      expect(lastAssistant).toBeDefined();
      const textBlocks = lastAssistant.blocks.filter(
        (b: any) => b.kind === "text",
      );
      const lastText = textBlocks[textBlocks.length - 1] as any;
      expect(lastText.text.length).toBeGreaterThan(0);
    });

    it("uses generic fallback when no content available", async () => {
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }),
          toolCall("echo", { text: "2" }),
        ],
        { maxIterations: 2 },
      );

      await agent.submitInput("test input");

      const session = agent.getSession();
      const assistantMsgs = session.messages.filter((m) => m.role === 2);
      const lastAssistant = assistantMsgs[assistantMsgs.length - 1];
      expect(lastAssistant).toBeDefined();
      const textBlocks = lastAssistant.blocks.filter(
        (b: any) => b.kind === "text",
      );
      const lastText = textBlocks[textBlocks.length - 1] as any;
      expect(lastText.text).toContain("wasn't able to complete");
    });
  });

  // ── empty response handling ───────────────────────────────────────

  describe("empty response handling", () => {
    it("continues on single empty response", async () => {
      const { agent } = createStreamingTestAgent(
        [empty(), text("real answer")],
        { maxIterations: 5 },
      );
      const collector = collectEvents(agent, "turn_complete");

      await agent.submitInput("test input");

      const payloads = collector.payloads<{ iterations: number }>(
        "turn_complete",
      );
      expect(payloads).toHaveLength(1);
      // Should have taken more than 1 iteration (empty then real)
      expect(payloads[0].iterations).toBeGreaterThanOrEqual(2);
    });

    it("bails after 3 consecutive empties", async () => {
      const { agent } = createStreamingTestAgent(
        [empty(), empty(), empty()],
        { maxIterations: 10 },
      );
      const collector = collectEvents(agent, "turn_complete");

      await agent.submitInput("test input");

      const payloads = collector.ofType("turn_complete");
      expect(payloads).toHaveLength(1);
    });
  });

  // ── usage tracking ────────────────────────────────────────────────

  describe("usage tracking", () => {
    it("emits cost_update events per iteration", async () => {
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }),
          text("done"),
        ],
        { maxIterations: 5 },
      );
      const collector = collectEvents(agent, "cost_update");

      await agent.submitInput("test input");

      const costUpdates = collector.ofType("cost_update");
      // At least one cost_update per LLM call
      expect(costUpdates.length).toBeGreaterThanOrEqual(1);
    });

    it("tracks cumulative token usage", async () => {
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }),
          toolCall("echo", { text: "2" }),
          text("done"),
        ],
        { maxIterations: 5 },
      );
      const collector = collectEvents(agent, "turn_complete");

      await agent.submitInput("test input");

      const payloads = collector.payloads<{
        usage: { promptTokens: number; completionTokens: number; totalTokens: number };
      }>("turn_complete");
      expect(payloads).toHaveLength(1);

      const usage = payloads[0].usage;
      expect(usage).toBeDefined();
      // Multiple iterations should accumulate tokens
      expect(usage.totalTokens).toBeGreaterThan(0);
      expect(usage.promptTokens).toBeGreaterThan(0);
    });
  });

  // ── interruption ──────────────────────────────────────────────────

  describe("interruption", () => {
    it("stops loop when interrupted during iteration", async () => {
      const { agent } = createStreamingTestAgent(
        [
          toolCall("echo", { text: "1" }),
          toolCall("echo", { text: "2" }),
          toolCall("echo", { text: "3" }),
          text("should not reach"),
        ],
        { maxIterations: 10 },
      );
      const collector = collectEvents(
        agent,
        "turn_complete",
        "interruption",
      );

      // Set interrupted flag directly before the loop progresses far
      // We listen for the first tool execution and immediately interrupt.
      agent.on("tool_execution_start", () => {
        // Call interrupt without await — it sets the flag synchronously
        agent.interrupt().catch(() => {});
      });

      await agent.submitInput("test input");

      // The agent should have stopped early — either via cancel phase
      // or by completing with fewer iterations than max.
      const turnPayloads = collector.payloads<{ iterations: number }>(
        "turn_complete",
      );
      // turn_complete or error should have fired
      expect(
        turnPayloads.length > 0 ||
          collector.ofType("interruption").length > 0,
      ).toBe(true);
    });
  });

  // ── middleware lifecycle ───────────────────────────────────────────

  describe("middleware lifecycle", () => {
    it("calls middleware hooks in correct order", async () => {
      const order: string[] = [];
      const mw = {
        onBeforeExecute: vi.fn(async () => {
          order.push("beforeExecute");
        }),
        onComplete: vi.fn(async () => {
          order.push("complete");
        }),
        onError: vi.fn(async () => {
          order.push("error");
        }),
      };

      const { agent } = createStreamingTestAgent([text("Hello")], {
        middleware: [mw],
      });

      await agent.submitInput("test input");

      expect(mw.onBeforeExecute).toHaveBeenCalled();
      expect(mw.onComplete).toHaveBeenCalled();
      expect(order.indexOf("beforeExecute")).toBeLessThan(
        order.indexOf("complete"),
      );
    });
  });

  // ── stream timeout ────────────────────────────────────────────────

  describe("stream timeout", () => {
    it("times out on hanging stream", async () => {
      const { agent } = createStreamingTestAgent([hang()], {
        streamTimeoutMs: 100,
      });
      const collector = collectEvents(agent, "error");

      await agent.submitInput("test input");

      const errors = collector.ofType("error");
      expect(errors.length).toBeGreaterThanOrEqual(1);
      const errorPayload = errors[0].payload as { message: string };
      expect(errorPayload.message).toContain("timed out");
    }, 10_000);
  });

  // ── finish_reason handling ────────────────────────────────────────

  describe("finish_reason handling", () => {
    it("continues on finish_reason=length", async () => {
      const { agent, remoteProvider } = createStreamingTestAgent(
        [
          text("partial response that was", "length"),
          text("complete answer"),
        ],
        { maxIterations: 5 },
      );
      const collector = collectEvents(agent, "turn_complete", "agent_thought");

      await agent.submitInput("test input");

      // The agent should have processed both scripts
      expect(remoteProvider.getCallIndex()).toBeGreaterThanOrEqual(2);

      // Check that a continuation message was injected into the request
      const requests = remoteProvider.getRequests();
      const lastRequest = requests[requests.length - 1];
      const continuationMsg = lastRequest.messages.find(
        (m: any) =>
          m.role === "user" &&
          typeof m.content === "string" &&
          m.content.includes("truncated"),
      );
      expect(continuationMsg).toBeDefined();
    });
  });
});
