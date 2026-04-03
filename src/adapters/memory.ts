import {
  MessageRole,
  type ConversationMessage,
  type ContentBlock as AsContentBlock,
  type Session,
} from "@sschepis/as-agent";
import type { ChatMessage, Role } from "@sschepis/lmscript";

const ROLE_TO_STRING: Record<MessageRole, Role> = {
  [MessageRole.System]: "system",
  [MessageRole.User]: "user",
  [MessageRole.Assistant]: "assistant",
  [MessageRole.Tool]: "user",
};

const STRING_TO_ROLE: Record<Role, MessageRole> = {
  system: MessageRole.System,
  user: MessageRole.User,
  assistant: MessageRole.Assistant,
};

/** Extract plain text from as-agent content blocks. */
function blocksToText(blocks: AsContentBlock[]): string {
  return blocks
    .map((b) => {
      switch (b.kind) {
        case "text":
          return b.text;
        case "tool_use":
          return `[Tool call: ${b.name}(${b.input})]`;
        case "tool_result":
          return b.isError
            ? `[Tool error (${b.toolName}): ${b.output}]`
            : `[Tool result (${b.toolName}): ${b.output}]`;
      }
    })
    .join("\n");
}

/** Convert an as-agent ConversationMessage to an lmscript ChatMessage. */
export function toChat(msg: ConversationMessage): ChatMessage {
  return {
    role: ROLE_TO_STRING[msg.role] ?? "user",
    content: blocksToText(msg.blocks),
  };
}

/** Convert an lmscript ChatMessage to an as-agent ConversationMessage. */
export function fromChat(msg: ChatMessage): ConversationMessage {
  const text = typeof msg.content === "string"
    ? msg.content
    : msg.content
        .filter((b) => b.type === "text")
        .map((b) => (b as { text: string }).text)
        .join("\n");

  return {
    role: STRING_TO_ROLE[msg.role] ?? MessageRole.User,
    blocks: [{ kind: "text", text }],
  };
}

/** Convert an entire as-agent Session to an array of lmscript ChatMessages. */
export function sessionToHistory(session: Session): ChatMessage[] {
  return session.messages.map(toChat);
}

/** Create an empty as-agent Session. */
export function createEmptySession(): Session {
  return { version: 1, messages: [] };
}
