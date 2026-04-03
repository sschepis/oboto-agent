import { describe, it, expect } from "vitest";
import { createTriageFunction, TriageSchema } from "../triage.js";

describe("createTriageFunction", () => {
  it("creates a valid LScriptFunction", () => {
    const fn = createTriageFunction("llama3:8b");

    expect(fn.name).toBe("triage");
    expect(fn.model).toBe("llama3:8b");
    expect(fn.temperature).toBe(0.1);
    expect(fn.maxRetries).toBe(1);
    expect(fn.system).toContain("triage classifier");
  });

  it("builds a prompt from input", () => {
    const fn = createTriageFunction("test-model");
    const prompt = fn.prompt({
      userInput: "What is 2+2?",
      recentContext: "user: hi\nassistant: hello",
      availableTools: "filesystem, terminal",
    });

    expect(prompt).toContain("What is 2+2?");
    expect(prompt).toContain("user: hi");
    expect(prompt).toContain("filesystem, terminal");
  });

  it("uses the provided model name", () => {
    const fn = createTriageFunction("my-custom-model");
    expect(fn.model).toBe("my-custom-model");
  });
});

describe("TriageSchema", () => {
  it("validates a valid escalation result", () => {
    const result = TriageSchema.parse({
      escalate: true,
      reasoning: "Needs code analysis",
    });
    expect(result.escalate).toBe(true);
    expect(result.reasoning).toBe("Needs code analysis");
    expect(result.directResponse).toBeUndefined();
  });

  it("validates a direct response result", () => {
    const result = TriageSchema.parse({
      escalate: false,
      reasoning: "Simple greeting",
      directResponse: "Hello!",
    });
    expect(result.escalate).toBe(false);
    expect(result.directResponse).toBe("Hello!");
  });

  it("rejects missing required fields", () => {
    expect(() => TriageSchema.parse({ escalate: true })).toThrow();
    expect(() => TriageSchema.parse({ reasoning: "test" })).toThrow();
  });

  it("rejects invalid types", () => {
    expect(() =>
      TriageSchema.parse({ escalate: "yes", reasoning: "test" })
    ).toThrow();
  });
});
