import { z } from "zod";
import { ContextStack, type LScriptRuntime, type ChatMessage, type LScriptFunction } from "@sschepis/lmscript";

const SummarySchema = z.object({
  summary: z.string().describe("A dense summary of the conversation so far"),
});

type SummaryInput = { conversation: string };

/**
 * Manages the sliding context window with automatic summarization.
 * Wraps lmscript's ContextStack and uses the local LLM for compression.
 */
export class ContextManager {
  private stack: ContextStack;
  private summarizeFn: LScriptFunction<SummaryInput, typeof SummarySchema>;

  constructor(
    private localRuntime: LScriptRuntime,
    localModelName: string,
    maxTokens: number
  ) {
    this.stack = new ContextStack({
      maxTokens,
      pruneStrategy: "summarize",
    });

    this.summarizeFn = {
      name: "summarize_context",
      model: localModelName,
      system:
        "You are a summarization engine. Compress the given conversation into a dense, factual summary that preserves all key information, decisions, and context needed for continued operation. Be concise but thorough.",
      prompt: ({ conversation }) => conversation,
      schema: SummarySchema,
      temperature: 0.2,
      maxRetries: 1,
    };

    this.stack.setSummarizer(async (messages: ChatMessage[]) => {
      const conversation = messages
        .map((m) => {
          const text =
            typeof m.content === "string"
              ? m.content
              : m.content
                  .filter((b) => b.type === "text")
                  .map((b) => (b as { text: string }).text)
                  .join(" ");
          return `${m.role}: ${text}`;
        })
        .join("\n");

      const result = await this.localRuntime.execute(this.summarizeFn, {
        conversation,
      });
      return result.data.summary;
    });
  }

  /** Append a message to the context. Triggers pruning if over budget. */
  async push(message: ChatMessage): Promise<void> {
    await this.stack.push(message);
  }

  /** Append multiple messages. */
  async pushAll(messages: ChatMessage[]): Promise<void> {
    await this.stack.pushAll(messages);
  }

  /** Get all messages in the current context window. */
  getMessages(): ChatMessage[] {
    return this.stack.getMessages();
  }

  /** Get estimated token count. */
  getTokenCount(): number {
    return this.stack.getTokenCount();
  }

  /** Clear all context. */
  clear(): void {
    this.stack.clear();
  }
}
