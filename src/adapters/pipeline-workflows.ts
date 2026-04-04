/**
 * pipeline-workflows.ts — Multi-step agent workflows using lmscript's Pipeline.
 *
 * This module provides pre-built and composable pipeline patterns for
 * chaining LLM calls where the typed output of one step feeds into the next.
 *
 * Key patterns:
 * 1. Triage → Execute → Summarize
 * 2. Analyze → Plan → Execute
 * 3. Custom pipeline builder for agent-specific workflows
 */

import { z } from "zod";
import {
  Pipeline,
  type LScriptRuntime,
  type LScriptFunction,
  type PipelineResult,
} from "@sschepis/lmscript";

// ── Schema definitions for pipeline steps ────────────────────────────

/** Output of a triage step — determines routing and intent. */
export const TriagePipelineSchema = z.object({
  intent: z.string().describe("The classified intent of the user input"),
  complexity: z.enum(["simple", "moderate", "complex"]).describe("Estimated task complexity"),
  requiresTools: z.boolean().describe("Whether tools are needed"),
  suggestedApproach: z.string().describe("Brief approach suggestion"),
  escalate: z.boolean().describe("Whether to escalate to the remote model"),
});
export type TriagePipelineOutput = z.infer<typeof TriagePipelineSchema>;

/** Output of a planning step — breaks work into steps. */
export const PlanSchema = z.object({
  steps: z.array(z.object({
    description: z.string(),
    toolRequired: z.string().optional(),
    expectedOutput: z.string().optional(),
  })).describe("Ordered list of steps to accomplish the task"),
  estimatedComplexity: z.enum(["low", "medium", "high"]),
  reasoning: z.string().describe("Why this plan was chosen"),
});
export type PlanOutput = z.infer<typeof PlanSchema>;

/** Output of an execution step — the actual work result. */
export const ExecutionSchema = z.object({
  response: z.string().describe("The response or result of executing the plan"),
  stepsCompleted: z.number().describe("Number of plan steps completed"),
  toolsUsed: z.array(z.string()).optional(),
  confidence: z.enum(["low", "medium", "high"]),
});
export type ExecutionOutput = z.infer<typeof ExecutionSchema>;

/** Output of a summarization step — condenses results. */
export const SummarySchema = z.object({
  summary: z.string().describe("Concise summary of what was accomplished"),
  keyPoints: z.array(z.string()).describe("Key points from the execution"),
  followUpSuggestions: z.array(z.string()).optional(),
});
export type SummaryOutput = z.infer<typeof SummarySchema>;

// ── Pipeline step factories ──────────────────────────────────────────

/**
 * Create a triage step function for the pipeline.
 */
export function createTriageStep(
  modelName: string,
  systemPrompt?: string,
): LScriptFunction<string, typeof TriagePipelineSchema> {
  return {
    name: "pipeline-triage",
    model: modelName,
    system: systemPrompt ?? "You are a task classifier. Analyze the input and determine its intent, complexity, and whether it requires tools or escalation.",
    prompt: (input: string) => `Classify this request:\n\n${input}`,
    schema: TriagePipelineSchema,
    temperature: 0.1,
    maxRetries: 1,
  };
}

/**
 * Create a planning step that takes triage output and produces a plan.
 */
export function createPlanStep(
  modelName: string,
  systemPrompt?: string,
): LScriptFunction<TriagePipelineOutput, typeof PlanSchema> {
  return {
    name: "pipeline-plan",
    model: modelName,
    system: systemPrompt ?? "You are a task planner. Given the triage analysis, create a step-by-step plan to accomplish the task.",
    prompt: (triage: TriagePipelineOutput) =>
      `Create an execution plan based on this analysis:\n` +
      `Intent: ${triage.intent}\n` +
      `Complexity: ${triage.complexity}\n` +
      `Requires tools: ${triage.requiresTools}\n` +
      `Suggested approach: ${triage.suggestedApproach}`,
    schema: PlanSchema,
    temperature: 0.3,
    maxRetries: 1,
  };
}

/**
 * Create an execution step that takes a plan and produces results.
 */
export function createExecutionStep(
  modelName: string,
  tools?: LScriptFunction<any, any>["tools"],
  systemPrompt?: string,
): LScriptFunction<PlanOutput, typeof ExecutionSchema> {
  return {
    name: "pipeline-execute",
    model: modelName,
    system: systemPrompt ?? "You are a task executor. Follow the given plan step by step and produce the result.",
    prompt: (plan: PlanOutput) => {
      const stepsStr = plan.steps
        .map((s, i) => `${i + 1}. ${s.description}${s.toolRequired ? ` (tool: ${s.toolRequired})` : ""}`)
        .join("\n");
      return `Execute this plan:\n\n${stepsStr}\n\nReasoning: ${plan.reasoning}`;
    },
    schema: ExecutionSchema,
    tools,
    temperature: 0.5,
    maxRetries: 1,
  };
}

/**
 * Create a summarization step that condenses execution results.
 */
export function createSummaryStep(
  modelName: string,
  systemPrompt?: string,
): LScriptFunction<ExecutionOutput, typeof SummarySchema> {
  return {
    name: "pipeline-summarize",
    model: modelName,
    system: systemPrompt ?? "You are a summarizer. Condense the execution results into a clear, concise summary.",
    prompt: (execution: ExecutionOutput) =>
      `Summarize these results:\n` +
      `Response: ${execution.response}\n` +
      `Steps completed: ${execution.stepsCompleted}\n` +
      `Confidence: ${execution.confidence}\n` +
      (execution.toolsUsed?.length ? `Tools used: ${execution.toolsUsed.join(", ")}` : ""),
    schema: SummarySchema,
    temperature: 0.3,
    maxRetries: 1,
  };
}

// ── Pre-built pipeline constructors ──────────────────────────────────

/**
 * Build a Triage → Plan → Execute pipeline.
 *
 * Takes raw user input, classifies it, plans the approach,
 * and executes the plan.
 *
 * ```ts
 * const pipeline = createTriagePlanExecutePipeline("gpt-4");
 * const result = await pipeline.execute(runtime, "Refactor the auth module");
 * console.log(result.finalData.response);
 * ```
 */
export function createTriagePlanExecutePipeline(
  modelName: string,
  tools?: LScriptFunction<any, any>["tools"],
): Pipeline<string, ExecutionOutput> {
  return Pipeline
    .from(createTriageStep(modelName))
    .pipe(createPlanStep(modelName))
    .pipe(createExecutionStep(modelName, tools));
}

/**
 * Build a Triage → Plan → Execute → Summarize pipeline.
 *
 * Full end-to-end pipeline that classifies, plans, executes,
 * and then summarizes the results.
 *
 * ```ts
 * const pipeline = createFullPipeline("gpt-4");
 * const result = await pipeline.execute(runtime, "Build a REST API");
 * console.log(result.finalData.summary);
 * console.log(`Total tokens: ${result.totalUsage.totalTokens}`);
 * ```
 */
export function createFullPipeline(
  modelName: string,
  tools?: LScriptFunction<any, any>["tools"],
): Pipeline<string, SummaryOutput> {
  return Pipeline
    .from(createTriageStep(modelName))
    .pipe(createPlanStep(modelName))
    .pipe(createExecutionStep(modelName, tools))
    .pipe(createSummaryStep(modelName));
}

/**
 * Build a simple Analyze → Respond pipeline.
 * Good for straightforward queries that don't need planning.
 */
export function createAnalyzeRespondPipeline(
  modelName: string,
): Pipeline<string, ExecutionOutput> {
  const analyzeStep: LScriptFunction<string, typeof TriagePipelineSchema> = {
    name: "pipeline-analyze",
    model: modelName,
    system: "You are an analyst. Understand the request and identify the best approach.",
    prompt: (input: string) => `Analyze this request:\n\n${input}`,
    schema: TriagePipelineSchema,
    temperature: 0.1,
    maxRetries: 1,
  };

  const respondStep: LScriptFunction<TriagePipelineOutput, typeof ExecutionSchema> = {
    name: "pipeline-respond",
    model: modelName,
    system: "You are a helpful assistant. Based on the analysis, provide a comprehensive response.",
    prompt: (analysis: TriagePipelineOutput) =>
      `Respond to this request:\n` +
      `Intent: ${analysis.intent}\n` +
      `Approach: ${analysis.suggestedApproach}`,
    schema: ExecutionSchema,
    temperature: 0.7,
    maxRetries: 1,
  };

  return Pipeline.from(analyzeStep).pipe(respondStep);
}

// ── Pipeline runner helper ───────────────────────────────────────────

/**
 * Configuration for running a pipeline within the agent context.
 */
export interface AgentPipelineConfig {
  /** The lmscript runtime to execute the pipeline on. */
  runtime: LScriptRuntime;
  /** The model name for all pipeline steps (can be overridden per-step). */
  modelName: string;
  /** Optional tools available to the execution step. */
  tools?: LScriptFunction<any, any>["tools"];
  /** Callback for each pipeline step completion. */
  onStepComplete?: (stepName: string, data: unknown, usage: { promptTokens: number; completionTokens: number; totalTokens: number } | undefined) => void;
}

/**
 * Run a pipeline and emit step completion events.
 * This is a convenience wrapper that adds observability.
 */
export async function runAgentPipeline<I, O>(
  pipeline: Pipeline<I, O>,
  input: I,
  config: AgentPipelineConfig,
): Promise<PipelineResult<O>> {
  const result = await pipeline.execute(config.runtime, input);

  if (config.onStepComplete) {
    for (const step of result.steps) {
      config.onStepComplete(step.name, step.data, step.usage);
    }
  }

  return result;
}
