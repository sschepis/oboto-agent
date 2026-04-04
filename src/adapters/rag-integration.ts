/**
 * rag-integration.ts — RAG pipeline integration with oboto-agent conversation history.
 *
 * This module bridges lmscript's RAG pipeline, embedding provider, and vector store
 * with the agent's conversation history (as-agent Sessions) and tool execution results.
 *
 * Key capabilities:
 * - Index conversation messages and tool results into a vector store
 * - Retrieve relevant past context when processing new user input
 * - Provide a RAG-augmented execution path for the agent
 */

import {
  RAGPipeline,
  MemoryVectorStore,
  type EmbeddingProvider,
  type VectorStore,
  type VectorSearchResult,
  type LScriptRuntime,
  type LScriptFunction,
  type ExecutionResult,
} from "@sschepis/lmscript";
import type { Session, ConversationMessage } from "@sschepis/as-agent";
import { MessageRole } from "@sschepis/as-agent";

// ── Types ─────────────────────────────────────────────────────────────

export interface ConversationRAGConfig {
  /** The embedding provider (e.g. OpenAI, local model) */
  embeddingProvider: EmbeddingProvider;

  /** Optional custom vector store. Defaults to MemoryVectorStore. */
  vectorStore?: VectorStore;

  /** Number of past context chunks to retrieve. Default: 5 */
  topK?: number;

  /** Minimum similarity score (0-1). Default: 0.3 */
  minScore?: number;

  /** Embedding model identifier. Default: provider's default */
  embeddingModel?: string;

  /**
   * Whether to automatically index conversation messages as they're added.
   * Default: true
   */
  autoIndex?: boolean;

  /**
   * Whether to index tool execution results.
   * Default: true
   */
  indexToolResults?: boolean;

  /** Maximum characters per indexed chunk. Longer messages are split. Default: 2000 */
  maxChunkSize?: number;

  /**
   * Custom context formatter for retrieved documents.
   * Default: numbered list with scores.
   */
  formatContext?: (results: VectorSearchResult[]) => string;
}

export interface RAGRetrievalResult {
  /** Retrieved context string ready for prompt injection */
  context: string;
  /** Raw search results */
  results: VectorSearchResult[];
  /** Number of documents in the store */
  totalDocuments: number;
}

// ── ConversationRAG ──────────────────────────────────────────────────

/**
 * ConversationRAG bridges lmscript's RAG pipeline with the agent's
 * conversation history and tool results.
 *
 * It maintains a vector store of conversation chunks and can retrieve
 * relevant past context for new queries. This is useful for:
 * - Long-running agent sessions where early context is pruned
 * - Multi-topic conversations where relevant past decisions need retrieval
 * - Tool result recall without re-execution
 *
 * ```ts
 * const rag = new ConversationRAG(runtime, {
 *   embeddingProvider: openaiEmbeddings,
 *   topK: 5,
 *   minScore: 0.3,
 * });
 *
 * // Index existing session history
 * await rag.indexSession(session);
 *
 * // Retrieve relevant context for a new query
 * const { context } = await rag.retrieve("What database schema did we discuss?");
 *
 * // Or use RAG-augmented execution directly
 * const result = await rag.executeWithContext(myFn, input, "search query");
 * ```
 */
export class ConversationRAG {
  private vectorStore: VectorStore;
  private embeddingProvider: EmbeddingProvider;
  private ragPipeline: RAGPipeline;
  private config: Required<ConversationRAGConfig>;
  private indexedMessageCount = 0;
  private runtime: LScriptRuntime;

  constructor(runtime: LScriptRuntime, config: ConversationRAGConfig) {
    this.runtime = runtime;
    this.vectorStore = config.vectorStore ?? new MemoryVectorStore();
    this.embeddingProvider = config.embeddingProvider;

    this.config = {
      embeddingProvider: config.embeddingProvider,
      vectorStore: this.vectorStore,
      topK: config.topK ?? 5,
      minScore: config.minScore ?? 0.3,
      embeddingModel: config.embeddingModel ?? "",
      autoIndex: config.autoIndex ?? true,
      indexToolResults: config.indexToolResults ?? true,
      maxChunkSize: config.maxChunkSize ?? 2000,
      formatContext: config.formatContext ?? defaultConversationContextFormatter,
    };

    this.ragPipeline = new RAGPipeline(runtime, {
      embeddingProvider: this.embeddingProvider,
      vectorStore: this.vectorStore,
      topK: this.config.topK,
      minScore: this.config.minScore,
      embeddingModel: this.config.embeddingModel,
      formatContext: this.config.formatContext,
    });
  }

  // ── Indexing ──────────────────────────────────────────────────────

  /**
   * Index an entire as-agent Session into the vector store.
   * Each message becomes one or more chunks (split by maxChunkSize).
   */
  async indexSession(session: Session): Promise<number> {
    const documents: Array<{ id: string; content: string; metadata?: Record<string, unknown> }> = [];

    for (let i = 0; i < session.messages.length; i++) {
      const msg = session.messages[i];
      const chunks = this.messageToChunks(msg, i);
      documents.push(...chunks);
    }

    if (documents.length > 0) {
      await this.ragPipeline.ingest(documents);
      this.indexedMessageCount += session.messages.length;
    }

    return documents.length;
  }

  /**
   * Index a single conversation message.
   * Call this after adding a message to the session for real-time indexing.
   */
  async indexMessage(msg: ConversationMessage, messageIndex?: number): Promise<void> {
    const idx = messageIndex ?? this.indexedMessageCount;
    const chunks = this.messageToChunks(msg, idx);

    if (chunks.length > 0) {
      await this.ragPipeline.ingest(chunks);
      this.indexedMessageCount++;
    }
  }

  /**
   * Index a tool execution result for later retrieval.
   */
  async indexToolResult(
    command: string,
    kwargs: Record<string, unknown>,
    result: string
  ): Promise<void> {
    if (!this.config.indexToolResults) return;

    const content = `Tool: ${command}\nArgs: ${JSON.stringify(kwargs)}\nResult: ${result}`;
    const chunks = this.splitChunks(content, `tool:${command}:${Date.now()}`);

    if (chunks.length > 0) {
      const documents = chunks.map((chunk, i) => ({
        id: chunk.id,
        content: chunk.content,
        metadata: {
          type: "tool_result",
          command,
          kwargs,
          chunkIndex: i,
          timestamp: Date.now(),
        },
      }));

      await this.ragPipeline.ingest(documents);
    }
  }

  // ── Retrieval ─────────────────────────────────────────────────────

  /**
   * Retrieve relevant past context for a query.
   * Returns formatted context string and raw results.
   */
  async retrieve(query: string): Promise<RAGRetrievalResult> {
    const [queryVector] = await this.embeddingProvider.embed(
      [query],
      this.config.embeddingModel || undefined
    );

    const results = await this.vectorStore.search(queryVector, this.config.topK);
    const filtered = results.filter(r => r.score >= this.config.minScore);
    const context = this.config.formatContext(filtered);
    const totalDocuments = await this.vectorStore.count();

    return { context, results: filtered, totalDocuments };
  }

  /**
   * Execute an lmscript function with RAG-augmented context from the
   * conversation history.
   *
   * This is the primary integration point — it uses lmscript's RAGPipeline
   * to inject relevant past conversation into the function's system prompt.
   */
  async executeWithContext<I, O extends import("zod").ZodType>(
    fn: LScriptFunction<I, O>,
    input: I,
    queryText?: string
  ): Promise<{ result: ExecutionResult<import("zod").infer<O>>; retrievedDocuments: VectorSearchResult[]; context: string }> {
    const ragResult = await this.ragPipeline.query(fn, input, queryText);
    return {
      result: ragResult.result as ExecutionResult<O>,
      retrievedDocuments: ragResult.retrievedDocuments,
      context: ragResult.context,
    };
  }

  // ── Utility ───────────────────────────────────────────────────────

  /** Get the number of indexed messages. */
  get messageCount(): number {
    return this.indexedMessageCount;
  }

  /** Get the total number of document chunks in the vector store. */
  async documentCount(): Promise<number> {
    return this.vectorStore.count();
  }

  /** Clear the vector store and reset counters. */
  async clear(): Promise<void> {
    await this.vectorStore.clear();
    this.indexedMessageCount = 0;
  }

  /** Get the underlying vector store (for advanced usage). */
  getVectorStore(): VectorStore {
    return this.vectorStore;
  }

  /** Get the underlying RAG pipeline (for advanced usage). */
  getRagPipeline(): RAGPipeline {
    return this.ragPipeline;
  }

  // ── Private ───────────────────────────────────────────────────────

  private messageToChunks(
    msg: ConversationMessage,
    messageIndex: number
  ): Array<{ id: string; content: string; metadata?: Record<string, unknown> }> {
    const roleLabel = messageRoleToLabel(msg.role);
    const text = blocksToText(msg.blocks);

    if (!text.trim()) return [];

    const prefixed = `[${roleLabel}]: ${text}`;
    const baseId = `msg:${messageIndex}`;

    return this.splitChunks(prefixed, baseId).map((chunk, i) => ({
      id: chunk.id,
      content: chunk.content,
      metadata: {
        type: "conversation",
        role: roleLabel,
        messageIndex,
        chunkIndex: i,
        timestamp: Date.now(),
      },
    }));
  }

  private splitChunks(
    text: string,
    baseId: string
  ): Array<{ id: string; content: string }> {
    const maxSize = this.config.maxChunkSize;

    if (text.length <= maxSize) {
      return [{ id: baseId, content: text }];
    }

    // Split on paragraph boundaries first, then sentence boundaries
    const chunks: Array<{ id: string; content: string }> = [];
    let remaining = text;
    let chunkIdx = 0;

    while (remaining.length > 0) {
      let splitAt = maxSize;

      if (remaining.length > maxSize) {
        // Try to split at paragraph boundary
        const paraIdx = remaining.lastIndexOf("\n\n", maxSize);
        if (paraIdx > maxSize * 0.3) {
          splitAt = paraIdx + 2;
        } else {
          // Try sentence boundary
          const sentIdx = remaining.lastIndexOf(". ", maxSize);
          if (sentIdx > maxSize * 0.3) {
            splitAt = sentIdx + 2;
          }
        }
      } else {
        splitAt = remaining.length;
      }

      chunks.push({
        id: `${baseId}:${chunkIdx}`,
        content: remaining.slice(0, splitAt).trim(),
      });

      remaining = remaining.slice(splitAt);
      chunkIdx++;
    }

    return chunks;
  }
}

// ── Helpers ──────────────────────────────────────────────────────────

function messageRoleToLabel(role: MessageRole): string {
  switch (role) {
    case MessageRole.System: return "system";
    case MessageRole.User: return "user";
    case MessageRole.Assistant: return "assistant";
    case MessageRole.Tool: return "tool";
    default: return "unknown";
  }
}

function blocksToText(blocks: ConversationMessage["blocks"]): string {
  return blocks
    .map((b) => {
      switch (b.kind) {
        case "text":
          return b.text;
        case "tool_use":
          return `[Tool: ${b.name}(${b.input})]`;
        case "tool_result":
          return b.isError
            ? `[Error: ${b.toolName}: ${b.output}]`
            : `[Result: ${b.toolName}: ${b.output}]`;
        default:
          return "";
      }
    })
    .join("\n");
}

/**
 * Default context formatter for conversation RAG results.
 * Shows role, score, and content for each retrieved chunk.
 */
function defaultConversationContextFormatter(results: VectorSearchResult[]): string {
  if (results.length === 0) return "";

  return [
    "## Relevant Past Context",
    "",
    ...results.map((r, i) => {
      const meta = r.document.metadata as Record<string, unknown> | undefined;
      const type = meta?.type ?? "unknown";
      const score = r.score.toFixed(3);
      return `[${i + 1}] (${type}, score: ${score})\n${r.document.content}`;
    }),
  ].join("\n");
}
