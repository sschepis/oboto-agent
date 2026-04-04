import { z } from "zod";
import type { LScriptFunction } from "@sschepis/lmscript";

/** Zod schema for structured triage output. */
export const TriageSchema = z.object({
  escalate: z
    .boolean()
    .describe("True if the request needs a powerful model, false if answerable directly"),
  reasoning: z
    .string()
    .describe("Brief explanation of the triage decision"),
  directResponse: z
    .string()
    .nullish()
    .transform(v => v ?? undefined)
    .describe("Direct answer if the request can be handled without escalation"),
});

export type TriageInput = {
  userInput: string;
  recentContext: string;
  availableTools: string;
};

const TRIAGE_SYSTEM = `You are a fast triage classifier for an AI agent system.
Your job is to decide whether a user's request can be answered directly (simple queries,
casual chat, short lookups) or needs to be escalated to a more powerful model
(complex reasoning, multi-step tool usage, code generation, analysis).

Rules:
- Only respond directly for truly trivial exchanges: greetings, thanks, or very simple factual questions.
- When responding directly, use a natural, warm, conversational tone. Do NOT list capabilities or describe what tools you have access to.
- If the user's message could benefit from tool usage or detailed reasoning, always escalate.
- If the request needs tool calls, code analysis, or multi-step reasoning: escalate.
- If unsure, escalate. It's better to over-escalate than to give a poor direct answer.
- Keep directResponse under 200 words when answering directly.
- Pay attention to the recent context — if the user is continuing a conversation, your response should reflect that context.

Respond with JSON matching the schema.`;

/**
 * Create an LScriptFunction for local-LLM triage classification.
 * The local model evaluates whether input needs escalation to the remote model.
 */
export function createTriageFunction(
  modelName: string
): LScriptFunction<TriageInput, typeof TriageSchema> {
  return {
    name: "triage",
    model: modelName,
    system: TRIAGE_SYSTEM,
    prompt: ({ userInput, recentContext, availableTools }) =>
      `Recent context:\n${recentContext}\n\nAvailable tools: ${availableTools}\n\nUser: ${userInput}`,
    schema: TriageSchema,
    temperature: 0.1,
    maxRetries: 1,
  };
}
