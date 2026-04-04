import { describe, it, expect, vi, beforeEach } from "vitest";
import { ConversationRAG } from "../adapters/rag-integration.js";
import { MessageRole } from "@sschepis/as-agent";
import type { Session, ConversationMessage } from "@sschepis/as-agent";
import type { EmbeddingProvider, VectorStore, VectorSearchResult } from "@sschepis/lmscript";
import { MemoryVectorStore, LScriptRuntime } from "@sschepis/lmscript";

// ── Mock embedding provider ──────────────────────────────────────────

class MockEmbeddingProvider implements EmbeddingProvider {
  readonly name = "mock-embedder";
  private callCount = 0;

  async embed(texts: string[], _model?: string): Promise<number[][]> {
    // Return deterministic fake embeddings that produce high cosine similarity
    // All embeddings point in roughly the same direction with slight variation
    return texts.map((text) => {
      this.callCount++;
      const seed = text.length % 10;
      // 384-dimensional embedding with high base + small perturbation
      return Array.from({ length: 384 }, (_, i) => 0.5 + Math.sin(seed + i * 0.01) * 0.1);
    });
  }

  getCallCount(): number {
    return this.callCount;
  }
}

// ── Mock LLM provider for runtime ────────────────────────────────────

function createMockRuntime(): LScriptRuntime {
  const mockProvider = {
    name: "mock",
    async chat() {
      return {
        content: '{"response": "mock"}',
        usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 },
      };
    },
  };

  return new LScriptRuntime({ provider: mockProvider });
}

// ── Tests ─────────────────────────────────────────────────────────────

describe("ConversationRAG", () => {
  let rag: ConversationRAG;
  let embedder: MockEmbeddingProvider;
  let runtime: LScriptRuntime;

  beforeEach(() => {
    embedder = new MockEmbeddingProvider();
    runtime = createMockRuntime();
    rag = new ConversationRAG(runtime, {
      embeddingProvider: embedder,
      topK: 3,
      minScore: 0.0, // Allow all results in tests
    });
  });

  describe("construction", () => {
    it("creates with default config", () => {
      expect(rag).toBeDefined();
      expect(rag.messageCount).toBe(0);
    });

    it("creates with custom vector store", () => {
      const customStore = new MemoryVectorStore();
      const customRag = new ConversationRAG(runtime, {
        embeddingProvider: embedder,
        vectorStore: customStore,
      });
      expect(customRag.getVectorStore()).toBe(customStore);
    });
  });

  describe("indexing", () => {
    it("indexes a single message", async () => {
      const msg: ConversationMessage = {
        role: MessageRole.User,
        blocks: [{ kind: "text", text: "Hello, how are you?" }],
      };

      await rag.indexMessage(msg, 0);
      expect(rag.messageCount).toBe(1);
      expect(await rag.documentCount()).toBeGreaterThan(0);
    });

    it("indexes an entire session", async () => {
      const session: Session = {
        version: 1,
        messages: [
          {
            role: MessageRole.User,
            blocks: [{ kind: "text", text: "What is TypeScript?" }],
          },
          {
            role: MessageRole.Assistant,
            blocks: [{ kind: "text", text: "TypeScript is a typed superset of JavaScript." }],
          },
          {
            role: MessageRole.User,
            blocks: [{ kind: "text", text: "How do I use generics?" }],
          },
        ],
      };

      const chunks = await rag.indexSession(session);
      expect(chunks).toBeGreaterThan(0);
      expect(rag.messageCount).toBe(3);
    });

    it("indexes tool results", async () => {
      await rag.indexToolResult("search", { query: "test" }, "Found 5 results");
      expect(await rag.documentCount()).toBeGreaterThan(0);
    });

    it("skips empty messages", async () => {
      const msg: ConversationMessage = {
        role: MessageRole.User,
        blocks: [{ kind: "text", text: "" }],
      };

      await rag.indexMessage(msg, 0);
      // Count won't increase because there's no text to index
      expect(await rag.documentCount()).toBe(0);
    });
  });

  describe("retrieval", () => {
    it("retrieves relevant context after indexing", async () => {
      const session: Session = {
        version: 1,
        messages: [
          {
            role: MessageRole.User,
            blocks: [{ kind: "text", text: "We need to implement a REST API for user management." }],
          },
          {
            role: MessageRole.Assistant,
            blocks: [{ kind: "text", text: "I'll design the API endpoints: GET /users, POST /users, PUT /users/:id, DELETE /users/:id." }],
          },
        ],
      };

      await rag.indexSession(session);

      const result = await rag.retrieve("What API endpoints did we plan?");
      expect(result.results.length).toBeGreaterThan(0);
      expect(result.totalDocuments).toBeGreaterThan(0);
    });

    it("returns empty context when nothing is indexed", async () => {
      const result = await rag.retrieve("anything");
      expect(result.results).toHaveLength(0);
      expect(result.context).toBe("");
    });
  });

  describe("clear", () => {
    it("resets the store and counters", async () => {
      const msg: ConversationMessage = {
        role: MessageRole.User,
        blocks: [{ kind: "text", text: "Test message for clearing" }],
      };

      await rag.indexMessage(msg, 0);
      expect(rag.messageCount).toBe(1);

      await rag.clear();
      expect(rag.messageCount).toBe(0);
      expect(await rag.documentCount()).toBe(0);
    });
  });

  describe("pipeline access", () => {
    it("exposes the vector store", () => {
      expect(rag.getVectorStore()).toBeDefined();
    });

    it("exposes the RAG pipeline", () => {
      expect(rag.getRagPipeline()).toBeDefined();
    });
  });
});
