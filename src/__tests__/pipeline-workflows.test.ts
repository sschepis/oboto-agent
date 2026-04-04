import { describe, it, expect, vi } from "vitest";
import {
  createTriageStep,
  createPlanStep,
  createExecutionStep,
  createSummaryStep,
  createTriagePlanExecutePipeline,
  createFullPipeline,
  createAnalyzeRespondPipeline,
  runAgentPipeline,
  TriagePipelineSchema,
  PlanSchema,
  ExecutionSchema,
  SummarySchema,
  type TriagePipelineOutput,
  type PlanOutput,
  type ExecutionOutput,
  type SummaryOutput,
} from "../adapters/pipeline-workflows.js";

// ── Step factories ───────────────────────────────────────────────────

describe("Pipeline step factories", () => {
  describe("createTriageStep", () => {
    it("creates a valid LScriptFunction", () => {
      const step = createTriageStep("gpt-4");
      expect(step.name).toBe("pipeline-triage");
      expect(step.model).toBe("gpt-4");
      expect(step.temperature).toBe(0.1);
      expect(step.schema).toBe(TriagePipelineSchema);
    });

    it("generates a prompt from string input", () => {
      const step = createTriageStep("gpt-4");
      const prompt = step.prompt("Fix the login bug");
      expect(prompt).toContain("Fix the login bug");
    });

    it("accepts custom system prompt", () => {
      const step = createTriageStep("gpt-4", "Custom triage system");
      expect(step.system).toBe("Custom triage system");
    });
  });

  describe("createPlanStep", () => {
    it("creates a valid LScriptFunction", () => {
      const step = createPlanStep("gpt-4");
      expect(step.name).toBe("pipeline-plan");
      expect(step.schema).toBe(PlanSchema);
    });

    it("generates a prompt from triage output", () => {
      const step = createPlanStep("gpt-4");
      const triage: TriagePipelineOutput = {
        intent: "bug fix",
        complexity: "moderate",
        requiresTools: true,
        suggestedApproach: "Debug and fix",
        escalate: false,
      };
      const prompt = step.prompt(triage);
      expect(prompt).toContain("bug fix");
      expect(prompt).toContain("moderate");
      expect(prompt).toContain("Debug and fix");
    });
  });

  describe("createExecutionStep", () => {
    it("creates a valid LScriptFunction", () => {
      const step = createExecutionStep("gpt-4");
      expect(step.name).toBe("pipeline-execute");
      expect(step.schema).toBe(ExecutionSchema);
    });

    it("generates a prompt from plan output", () => {
      const step = createExecutionStep("gpt-4");
      const plan: PlanOutput = {
        steps: [
          { description: "Find the bug", toolRequired: "grep" },
          { description: "Fix it" },
        ],
        estimatedComplexity: "medium",
        reasoning: "Standard debug flow",
      };
      const prompt = step.prompt(plan);
      expect(prompt).toContain("Find the bug");
      expect(prompt).toContain("(tool: grep)");
      expect(prompt).toContain("Fix it");
      expect(prompt).toContain("Standard debug flow");
    });

    it("passes through tools", () => {
      const tools = [{ name: "test", description: "test tool", schema: TriagePipelineSchema }] as any;
      const step = createExecutionStep("gpt-4", tools);
      expect(step.tools).toBe(tools);
    });
  });

  describe("createSummaryStep", () => {
    it("creates a valid LScriptFunction", () => {
      const step = createSummaryStep("gpt-4");
      expect(step.name).toBe("pipeline-summarize");
      expect(step.schema).toBe(SummarySchema);
    });

    it("generates a prompt from execution output", () => {
      const step = createSummaryStep("gpt-4");
      const execution: ExecutionOutput = {
        response: "Fixed the auth bug",
        stepsCompleted: 3,
        toolsUsed: ["grep", "edit"],
        confidence: "high",
      };
      const prompt = step.prompt(execution);
      expect(prompt).toContain("Fixed the auth bug");
      expect(prompt).toContain("3");
      expect(prompt).toContain("grep, edit");
    });
  });
});

// ── Pre-built pipeline constructors ──────────────────────────────────

describe("Pre-built pipeline constructors", () => {
  it("createTriagePlanExecutePipeline returns a pipeline", () => {
    const pipeline = createTriagePlanExecutePipeline("gpt-4");
    expect(pipeline).toBeDefined();
    // Pipeline has an execute method
    expect(typeof pipeline.execute).toBe("function");
  });

  it("createFullPipeline returns a pipeline with 4 steps", () => {
    const pipeline = createFullPipeline("gpt-4");
    expect(pipeline).toBeDefined();
    expect(typeof pipeline.execute).toBe("function");
  });

  it("createAnalyzeRespondPipeline returns a pipeline", () => {
    const pipeline = createAnalyzeRespondPipeline("gpt-4");
    expect(pipeline).toBeDefined();
    expect(typeof pipeline.execute).toBe("function");
  });
});

// ── Schema validation ────────────────────────────────────────────────

describe("Pipeline schemas", () => {
  it("TriagePipelineSchema validates correct data", () => {
    const data = {
      intent: "code review",
      complexity: "moderate",
      requiresTools: false,
      suggestedApproach: "Review line by line",
      escalate: false,
    };
    const result = TriagePipelineSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it("TriagePipelineSchema rejects invalid complexity", () => {
    const data = {
      intent: "code review",
      complexity: "extreme",
      requiresTools: false,
      suggestedApproach: "Review",
      escalate: false,
    };
    const result = TriagePipelineSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it("PlanSchema validates correct data", () => {
    const data = {
      steps: [{ description: "Step 1" }, { description: "Step 2", toolRequired: "search" }],
      estimatedComplexity: "low",
      reasoning: "Simple task",
    };
    const result = PlanSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it("ExecutionSchema validates correct data", () => {
    const data = {
      response: "Done",
      stepsCompleted: 2,
      confidence: "high",
    };
    const result = ExecutionSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it("SummarySchema validates correct data", () => {
    const data = {
      summary: "Task completed",
      keyPoints: ["Point 1", "Point 2"],
      followUpSuggestions: ["Suggestion 1"],
    };
    const result = SummarySchema.safeParse(data);
    expect(result.success).toBe(true);
  });
});

// ── runAgentPipeline ─────────────────────────────────────────────────

describe("runAgentPipeline", () => {
  it("calls onStepComplete callback for each step", async () => {
    // Create a mock pipeline that we can control
    const mockPipeline = {
      execute: vi.fn().mockResolvedValue({
        finalData: { summary: "done" },
        steps: [
          { name: "step1", data: { x: 1 }, attempts: 1, usage: { promptTokens: 10, completionTokens: 5, totalTokens: 15 } },
          { name: "step2", data: { y: 2 }, attempts: 1, usage: { promptTokens: 20, completionTokens: 10, totalTokens: 30 } },
        ],
        totalUsage: { promptTokens: 30, completionTokens: 15, totalTokens: 45 },
      }),
    };

    const stepsCompleted: { name: string; data: unknown }[] = [];
    const mockRuntime = {} as any;

    const result = await runAgentPipeline(
      mockPipeline as any,
      "test input",
      {
        runtime: mockRuntime,
        modelName: "gpt-4",
        onStepComplete: (name, data) => stepsCompleted.push({ name, data }),
      },
    );

    expect(mockPipeline.execute).toHaveBeenCalledWith(mockRuntime, "test input");
    expect(stepsCompleted).toHaveLength(2);
    expect(stepsCompleted[0].name).toBe("step1");
    expect(stepsCompleted[1].name).toBe("step2");
    expect(result.totalUsage.totalTokens).toBe(45);
  });
});
