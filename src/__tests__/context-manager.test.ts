import { describe, it, expect, vi } from "vitest";
import { ContextManager } from "../context-manager.js";
import type { ChatMessage } from "@sschepis/lmscript";

function makeMockRuntime(overrides: Record<string, unknown> = {}) {
  return {
    execute: vi.fn().mockResolvedValue({
      data: { summary: "Summarized conversation" },
    }),
    ...overrides,
  } as any;
}

function msg(role: "user" | "assistant", content: string): ChatMessage {
  return { role, content };
}

describe("ContextManager", () => {
  // ---------------------------------------------------------------
  // 1. Construction
  // ---------------------------------------------------------------
  describe("construction", () => {
    it("creates with valid parameters and initializes empty", () => {
      const runtime = makeMockRuntime();
      const cm = new ContextManager(runtime, "test-model", 4096);

      expect(cm.getMessages()).toEqual([]);
      expect(cm.getTokenCount()).toBe(0);
    });
  });

  // ---------------------------------------------------------------
  // 2. push
  // ---------------------------------------------------------------
  describe("push", () => {
    it("adds a single message and getMessages returns it", async () => {
      const runtime = makeMockRuntime();
      const cm = new ContextManager(runtime, "test-model", 4096);

      await cm.push(msg("user", "Hello world"));

      const messages = cm.getMessages();
      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe("user");
      expect(messages[0].content).toBe("Hello world");
    });
  });

  // ---------------------------------------------------------------
  // 3. pushAll
  // ---------------------------------------------------------------
  describe("pushAll", () => {
    it("adds multiple messages and getMessages returns all in order", async () => {
      const runtime = makeMockRuntime();
      const cm = new ContextManager(runtime, "test-model", 4096);

      const batch: ChatMessage[] = [
        msg("user", "First message"),
        msg("assistant", "Second message"),
        msg("user", "Third message"),
      ];

      await cm.pushAll(batch);

      const messages = cm.getMessages();
      expect(messages).toHaveLength(3);
      expect(messages[0].content).toBe("First message");
      expect(messages[1].content).toBe("Second message");
      expect(messages[2].content).toBe("Third message");
    });
  });

  // ---------------------------------------------------------------
  // 4. getTokenCount
  // ---------------------------------------------------------------
  describe("getTokenCount", () => {
    it("returns non-zero after pushing messages", async () => {
      const runtime = makeMockRuntime();
      const cm = new ContextManager(runtime, "test-model", 4096);

      await cm.push(msg("user", "Hello world"));

      expect(cm.getTokenCount()).toBeGreaterThan(0);
    });
  });

  // ---------------------------------------------------------------
  // 5. clear
  // ---------------------------------------------------------------
  describe("clear", () => {
    it("removes all messages and resets token count to 0", async () => {
      const runtime = makeMockRuntime();
      const cm = new ContextManager(runtime, "test-model", 4096);

      await cm.push(msg("user", "Hello world"));
      await cm.push(msg("assistant", "Hi there"));
      expect(cm.getMessages().length).toBeGreaterThan(0);
      expect(cm.getTokenCount()).toBeGreaterThan(0);

      cm.clear();

      expect(cm.getMessages()).toEqual([]);
      expect(cm.getTokenCount()).toBe(0);
    });
  });

  // ---------------------------------------------------------------
  // 6. Summarization triggers when token budget is exceeded
  // ---------------------------------------------------------------
  describe("summarization", () => {
    it("calls the summarizer when token budget is exceeded", async () => {
      const runtime = makeMockRuntime();
      // maxTokens=50 means ~200 chars before pruning triggers
      const cm = new ContextManager(runtime, "test-model", 50);

      // Each message is ~50+ chars, so 5 messages should exceed the budget
      const longContent = "A".repeat(60);
      for (let i = 0; i < 5; i++) {
        await cm.push(msg(i % 2 === 0 ? "user" : "assistant", `${longContent} ${i}`));
      }

      // The runtime.execute should have been called at least once for summarization
      expect(runtime.execute).toHaveBeenCalled();

      // After pruning, messages should still exist (including the summary)
      const messages = cm.getMessages();
      expect(messages.length).toBeGreaterThan(0);

      // The summary text should appear somewhere in the remaining messages
      const allContent = messages
        .map((m) => (typeof m.content === "string" ? m.content : ""))
        .join(" ");
      expect(allContent).toContain("Summarized conversation");
    });
  });

  // ---------------------------------------------------------------
  // 7. Summarization fallback on runtime failure
  // ---------------------------------------------------------------
  describe("summarization fallback", () => {
    it("falls back to truncation when runtime.execute rejects", async () => {
      const runtime = makeMockRuntime({
        execute: vi.fn().mockRejectedValue(new Error("LLM unavailable")),
      });
      const cm = new ContextManager(runtime, "test-model", 50);

      // Suppress console.warn for this test
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const longContent = "B".repeat(60);
      // Push enough messages to trigger pruning; should not throw
      for (let i = 0; i < 5; i++) {
        await cm.push(msg(i % 2 === 0 ? "user" : "assistant", `${longContent} ${i}`));
      }

      // No crash, messages still exist
      const messages = cm.getMessages();
      expect(messages.length).toBeGreaterThan(0);

      // console.warn should have been called with the fallback message
      expect(warnSpy).toHaveBeenCalled();
      const warnCalls = warnSpy.mock.calls.flat().join(" ");
      expect(warnCalls).toContain("Summarization failed");

      warnSpy.mockRestore();
    });
  });

  // ---------------------------------------------------------------
  // 8. Multiple pruning cycles keep context bounded
  // ---------------------------------------------------------------
  describe("multiple pruning cycles", () => {
    it("keeps the context bounded after many messages", async () => {
      const runtime = makeMockRuntime();
      const maxTokens = 50; // ~200 chars
      const cm = new ContextManager(runtime, "test-model", maxTokens);

      const longContent = "C".repeat(60);
      // Push 20 messages in sequence to trigger multiple pruning cycles
      for (let i = 0; i < 20; i++) {
        await cm.push(msg(i % 2 === 0 ? "user" : "assistant", `${longContent} msg-${i}`));
      }

      // Token count should stay at or below the budget (with some tolerance
      // for the last unpruned message)
      const tokenCount = cm.getTokenCount();
      // Allow generous headroom: the stack may temporarily exceed the limit
      // by one message before the next prune, but it should not grow unbounded.
      expect(tokenCount).toBeLessThan(maxTokens * 3);

      // The summarizer should have been called multiple times
      expect(runtime.execute.mock.calls.length).toBeGreaterThanOrEqual(2);
    });
  });
});
