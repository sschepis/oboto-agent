import { describe, it, expect } from "vitest";
import { MessageRole } from "@sschepis/as-agent";
import type { ConversationMessage, Session } from "@sschepis/as-agent";
import type { ChatMessage } from "@sschepis/lmscript";
import { toChat, fromChat, sessionToHistory, createEmptySession } from "../adapters/memory.js";

describe("memory adapter", () => {
  describe("toChat", () => {
    it("converts a text message", () => {
      const msg: ConversationMessage = {
        role: MessageRole.User,
        blocks: [{ kind: "text", text: "Hello world" }],
      };
      const chat = toChat(msg);
      expect(chat.role).toBe("user");
      expect(chat.content).toBe("Hello world");
    });

    it("converts an assistant message", () => {
      const msg: ConversationMessage = {
        role: MessageRole.Assistant,
        blocks: [{ kind: "text", text: "Hi there" }],
      };
      const chat = toChat(msg);
      expect(chat.role).toBe("assistant");
      expect(chat.content).toBe("Hi there");
    });

    it("converts a system message", () => {
      const msg: ConversationMessage = {
        role: MessageRole.System,
        blocks: [{ kind: "text", text: "You are helpful." }],
      };
      const chat = toChat(msg);
      expect(chat.role).toBe("system");
    });

    it("converts tool role to user", () => {
      const msg: ConversationMessage = {
        role: MessageRole.Tool,
        blocks: [
          {
            kind: "tool_result",
            toolUseId: "t1",
            toolName: "fs_read",
            output: "file contents",
            isError: false,
          },
        ],
      };
      const chat = toChat(msg);
      expect(chat.role).toBe("user");
      expect(chat.content).toContain("[Tool result (fs_read): file contents]");
    });

    it("formats tool error blocks", () => {
      const msg: ConversationMessage = {
        role: MessageRole.Tool,
        blocks: [
          {
            kind: "tool_result",
            toolUseId: "t1",
            toolName: "bash",
            output: "command not found",
            isError: true,
          },
        ],
      };
      const chat = toChat(msg);
      expect(chat.content).toContain("[Tool error (bash): command not found]");
    });

    it("formats tool_use blocks", () => {
      const msg: ConversationMessage = {
        role: MessageRole.Assistant,
        blocks: [
          {
            kind: "tool_use",
            id: "t1",
            name: "filesystem",
            input: '{"path": "/tmp"}',
          },
        ],
      };
      const chat = toChat(msg);
      expect(chat.content).toContain('[Tool call: filesystem({"path": "/tmp"})]');
    });

    it("joins multiple blocks with newlines", () => {
      const msg: ConversationMessage = {
        role: MessageRole.User,
        blocks: [
          { kind: "text", text: "Line 1" },
          { kind: "text", text: "Line 2" },
        ],
      };
      const chat = toChat(msg);
      expect(chat.content).toBe("Line 1\nLine 2");
    });
  });

  describe("fromChat", () => {
    it("converts a string content message", () => {
      const chat: ChatMessage = { role: "user", content: "Hello" };
      const msg = fromChat(chat);
      expect(msg.role).toBe(MessageRole.User);
      expect(msg.blocks).toEqual([{ kind: "text", text: "Hello" }]);
    });

    it("converts an assistant message", () => {
      const chat: ChatMessage = { role: "assistant", content: "Response" };
      const msg = fromChat(chat);
      expect(msg.role).toBe(MessageRole.Assistant);
    });

    it("converts a system message", () => {
      const chat: ChatMessage = { role: "system", content: "You are..." };
      const msg = fromChat(chat);
      expect(msg.role).toBe(MessageRole.System);
    });

    it("handles content block array", () => {
      const chat: ChatMessage = {
        role: "user",
        content: [
          { type: "text", text: "Part 1" },
          { type: "text", text: "Part 2" },
        ],
      };
      const msg = fromChat(chat);
      expect(msg.blocks[0]).toEqual({ kind: "text", text: "Part 1\nPart 2" });
    });
  });

  describe("sessionToHistory", () => {
    it("converts an entire session", () => {
      const session: Session = {
        version: 1,
        messages: [
          {
            role: MessageRole.System,
            blocks: [{ kind: "text", text: "System prompt" }],
          },
          {
            role: MessageRole.User,
            blocks: [{ kind: "text", text: "Hi" }],
          },
          {
            role: MessageRole.Assistant,
            blocks: [{ kind: "text", text: "Hello!" }],
          },
        ],
      };
      const history = sessionToHistory(session);
      expect(history).toHaveLength(3);
      expect(history[0].role).toBe("system");
      expect(history[1].role).toBe("user");
      expect(history[2].role).toBe("assistant");
    });

    it("handles empty session", () => {
      const session: Session = { version: 1, messages: [] };
      expect(sessionToHistory(session)).toEqual([]);
    });
  });

  describe("createEmptySession", () => {
    it("creates a session with version 1 and no messages", () => {
      const session = createEmptySession();
      expect(session.version).toBe(1);
      expect(session.messages).toEqual([]);
    });
  });
});
