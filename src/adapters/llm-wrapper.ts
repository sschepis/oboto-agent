import type { LLMProvider, LLMRequest, LLMResponse } from "@sschepis/lmscript";
import type {
  BaseProvider,
  StandardChatParams,
  Message,
  ToolDefinition as WrapperToolDef,
} from "@sschepis/llm-wrapper";

/**
 * Adapt a llm-wrapper BaseProvider into lmscript's LLMProvider interface.
 *
 * This allows lmscript's LScriptRuntime to use llm-wrapper providers for
 * structured calls (e.g. triage) that need schema validation.
 */
export function toLmscriptProvider(
  provider: BaseProvider,
  name?: string
): LLMProvider {
  return {
    name: name ?? provider.providerName,

    async chat(request: LLMRequest): Promise<LLMResponse> {
      // Convert lmscript messages → llm-wrapper messages
      const messages: Message[] = request.messages.map((m) => ({
        role: m.role,
        content:
          typeof m.content === "string"
            ? m.content
            : m.content
                .filter((b) => b.type === "text")
                .map((b) => (b as { type: "text"; text: string }).text)
                .join("\n"),
      }));

      // Convert lmscript tools → llm-wrapper tool definitions
      let tools: WrapperToolDef[] | undefined;
      if (request.tools && request.tools.length > 0) {
        tools = request.tools.map((t) => ({
          type: "function" as const,
          function: {
            name: t.name,
            description: t.description,
            parameters: t.parameters as Record<string, unknown>,
          },
        }));
      }

      const params: StandardChatParams = {
        model: request.model,
        messages,
        temperature: request.temperature,
        ...(tools ? { tools } : {}),
        ...(request.jsonMode
          ? { response_format: { type: "json_object" as const } }
          : {}),
      };

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
  };
}
