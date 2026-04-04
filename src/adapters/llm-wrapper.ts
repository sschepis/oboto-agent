import type { LLMProvider, LLMRequest, LLMResponse } from "@sschepis/lmscript";
import type {
  BaseProvider,
  StandardChatParams,
  Message,
  ToolDefinition as WrapperToolDef,
} from "@sschepis/llm-wrapper";
import type { LLMRouter } from "@sschepis/llm-wrapper";

/**
 * A provider-like object that has `chat()` and `stream()` methods.
 * Both `BaseProvider` and `LLMRouter` implement this interface.
 */
type ProviderLike = {
  chat(params: StandardChatParams): Promise<import("@sschepis/llm-wrapper").StandardChatResponse>;
  stream(params: StandardChatParams): AsyncIterable<import("@sschepis/llm-wrapper").StandardChatChunk>;
  readonly providerName?: string;
};

/**
 * Convert lmscript messages to llm-wrapper messages.
 */
function convertMessages(messages: LLMRequest["messages"]): Message[] {
  return messages.map((m) => ({
    role: m.role,
    content:
      typeof m.content === "string"
        ? m.content
        : m.content
            .filter((b) => b.type === "text")
            .map((b) => (b as { type: "text"; text: string }).text)
            .join("\n"),
  }));
}

/**
 * Convert lmscript tool format to llm-wrapper tool definitions.
 */
function convertTools(tools?: LLMRequest["tools"]): WrapperToolDef[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters as Record<string, unknown>,
    },
  }));
}

/**
 * Build llm-wrapper StandardChatParams from an lmscript LLMRequest.
 */
function buildParams(request: LLMRequest, tools?: WrapperToolDef[]): StandardChatParams {
  return {
    model: request.model,
    messages: convertMessages(request.messages),
    temperature: request.temperature,
    ...(tools ? { tools } : {}),
    ...(request.jsonMode
      ? { response_format: { type: "json_object" as const } }
      : {}),
  };
}

/**
 * Adapt a llm-wrapper BaseProvider (or LLMRouter) into lmscript's LLMProvider interface.
 *
 * This allows lmscript's LScriptRuntime to use llm-wrapper providers for
 * structured calls (e.g. triage) that need schema validation.
 *
 * Bridges both `chat()` and `chatStream()` — enabling streaming through the
 * full lmscript stack (including `executeStream()`).
 */
export function toLmscriptProvider(
  provider: ProviderLike,
  name?: string
): LLMProvider {
  return {
    name: name ?? (provider as BaseProvider).providerName ?? "llm-wrapper",

    async chat(request: LLMRequest): Promise<LLMResponse> {
      const tools = convertTools(request.tools);
      const params = buildParams(request, tools);

      const response = await provider.chat(params);
      const choice = response.choices[0];

      // Convert tool calls back to lmscript format
      let toolCalls: LLMResponse["toolCalls"];
      if (choice?.message?.tool_calls) {
        toolCalls = choice.message.tool_calls.map((tc) => ({
          id: tc.id,
          name: tc.function.name,
          arguments: tc.function.arguments,
        }));
      }

      return {
        content: (choice?.message?.content as string) ?? "",
        usage: response.usage
          ? {
              promptTokens: response.usage.prompt_tokens,
              completionTokens: response.usage.completion_tokens,
              totalTokens: response.usage.total_tokens,
            }
          : undefined,
        toolCalls,
      };
    },

    async *chatStream(request: LLMRequest): AsyncIterable<string> {
      const tools = convertTools(request.tools);
      const params = buildParams(request, tools);

      for await (const chunk of provider.stream({ ...params, stream: true })) {
        const delta = chunk.choices?.[0]?.delta;
        if (delta?.content) {
          yield delta.content;
        }
      }
    },
  };
}
