import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  UsageBridge,
  asTokenUsageToLmscript,
  lmscriptToAsTokenUsage,
  estimateCostFromAsAgent,
  type UnifiedCostSummary,
} from "../adapters/usage-bridge.js";
import { AgentUsageTracker } from "../adapters/as-agent-features.js";
import type { TokenUsage, ModelPricing as AsModelPricing, UsageCostEstimate } from "@sschepis/as-agent";
import type { CostTracker, ModelPricing as LmModelPricing } from "@sschepis/lmscript";

// ── Mock factories ────────────────────────────────────────────────────

function createMockCostTracker(): CostTracker {
  const usageMap = new Map<string, { calls: number; totalTokens: number; promptTokens: number; completionTokens: number }>();
  let totalTokensAccum = 0;

  return {
    trackUsage: vi.fn((fnName: string, usage: { promptTokens: number; completionTokens: number; totalTokens: number }) => {
      totalTokensAccum += usage.totalTokens;
      const existing = usageMap.get(fnName) ?? { calls: 0, totalTokens: 0, promptTokens: 0, completionTokens: 0 };
      usageMap.set(fnName, {
        calls: existing.calls + 1,
        totalTokens: existing.totalTokens + usage.totalTokens,
        promptTokens: existing.promptTokens + usage.promptTokens,
        completionTokens: existing.completionTokens + usage.completionTokens,
      });
    }),
    getTotalTokens: vi.fn(() => totalTokensAccum),
    getTotalCost: vi.fn((pricing?: LmModelPricing) => {
      if (!pricing) return 0;
      // Simplified cost calc for testing
      return totalTokensAccum * 0.001;
    }),
    getUsageByFunction: vi.fn(() => usageMap),
    checkBudget: vi.fn(),
    reset: vi.fn(() => {
      usageMap.clear();
      totalTokensAccum = 0;
    }),
  } as unknown as CostTracker;
}

function makeAsUsage(input: number, output: number, cacheCreate = 0, cacheRead = 0): TokenUsage {
  return {
    inputTokens: input,
    outputTokens: output,
    cacheCreationInputTokens: cacheCreate,
    cacheReadInputTokens: cacheRead,
  };
}

function makeAsPricing(): AsModelPricing {
  return {
    inputCostPerMillion: 3.0,
    outputCostPerMillion: 15.0,
    cacheCreationCostPerMillion: 3.75,
    cacheReadCostPerMillion: 0.3,
  };
}

// ── asTokenUsageToLmscript ───────────────────────────────────────────

describe("asTokenUsageToLmscript", () => {
  it("converts basic token counts", () => {
    const result = asTokenUsageToLmscript(makeAsUsage(100, 50));
    expect(result.promptTokens).toBe(100);
    expect(result.completionTokens).toBe(50);
    expect(result.totalTokens).toBe(150);
  });

  it("includes cache tokens in total", () => {
    const result = asTokenUsageToLmscript(makeAsUsage(100, 50, 20, 10));
    expect(result.promptTokens).toBe(100);
    expect(result.completionTokens).toBe(50);
    expect(result.totalTokens).toBe(180); // 100 + 50 + 20 + 10
  });

  it("handles zero usage", () => {
    const result = asTokenUsageToLmscript(makeAsUsage(0, 0, 0, 0));
    expect(result.promptTokens).toBe(0);
    expect(result.completionTokens).toBe(0);
    expect(result.totalTokens).toBe(0);
  });
});

// ── lmscriptToAsTokenUsage ───────────────────────────────────────────

describe("lmscriptToAsTokenUsage", () => {
  it("converts lmscript format to as-agent format", () => {
    const result = lmscriptToAsTokenUsage({
      promptTokens: 200,
      completionTokens: 100,
      totalTokens: 300,
    });
    expect(result.inputTokens).toBe(200);
    expect(result.outputTokens).toBe(100);
    expect(result.cacheCreationInputTokens).toBe(0);
    expect(result.cacheReadInputTokens).toBe(0);
  });

  it("sets cache tokens to 0 (lmscript has no cache concept)", () => {
    const result = lmscriptToAsTokenUsage({
      promptTokens: 500,
      completionTokens: 250,
      totalTokens: 750,
    });
    expect(result.cacheCreationInputTokens).toBe(0);
    expect(result.cacheReadInputTokens).toBe(0);
  });

  it("handles zero usage", () => {
    const result = lmscriptToAsTokenUsage({
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    });
    expect(result.inputTokens).toBe(0);
    expect(result.outputTokens).toBe(0);
  });
});

// ── estimateCostFromAsAgent ──────────────────────────────────────────

describe("estimateCostFromAsAgent", () => {
  it("calculates costs from per-million-token pricing", () => {
    const usage = makeAsUsage(1_000_000, 500_000);
    const pricing = makeAsPricing();
    const cost = estimateCostFromAsAgent(usage, pricing);

    expect(cost.inputCostUsd).toBeCloseTo(3.0);     // 1M * 3.0/M
    expect(cost.outputCostUsd).toBeCloseTo(7.5);     // 0.5M * 15.0/M
    expect(cost.cacheCreationCostUsd).toBeCloseTo(0); // 0 tokens
    expect(cost.cacheReadCostUsd).toBeCloseTo(0);     // 0 tokens
    expect(cost.totalCostUsd).toBeCloseTo(10.5);
  });

  it("includes cache costs", () => {
    const usage = makeAsUsage(100_000, 50_000, 200_000, 300_000);
    const pricing = makeAsPricing();
    const cost = estimateCostFromAsAgent(usage, pricing);

    expect(cost.inputCostUsd).toBeCloseTo(0.3);           // 0.1M * 3.0
    expect(cost.outputCostUsd).toBeCloseTo(0.75);          // 0.05M * 15.0
    expect(cost.cacheCreationCostUsd).toBeCloseTo(0.75);   // 0.2M * 3.75
    expect(cost.cacheReadCostUsd).toBeCloseTo(0.09);       // 0.3M * 0.3
    expect(cost.totalCostUsd).toBeCloseTo(1.89);
  });

  it("returns zero costs for zero usage", () => {
    const usage = makeAsUsage(0, 0, 0, 0);
    const pricing = makeAsPricing();
    const cost = estimateCostFromAsAgent(usage, pricing);

    expect(cost.totalCostUsd).toBe(0);
    expect(cost.inputCostUsd).toBe(0);
    expect(cost.outputCostUsd).toBe(0);
  });
});

// ── UsageBridge ──────────────────────────────────────────────────────

describe("UsageBridge", () => {
  let tracker: AgentUsageTracker;
  let costTracker: CostTracker;
  let bridge: UsageBridge;

  beforeEach(() => {
    tracker = new AgentUsageTracker();
    costTracker = createMockCostTracker();
    bridge = new UsageBridge(tracker, costTracker);
  });

  describe("recordFromLmscript", () => {
    it("records usage in both trackers", () => {
      bridge.recordFromLmscript("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });

      // Check as-agent tracker
      const asUsage = tracker.currentTurnUsage();
      expect(asUsage.inputTokens).toBe(100);
      expect(asUsage.outputTokens).toBe(50);

      // Check lmscript cost tracker
      expect(costTracker.trackUsage).toHaveBeenCalledWith("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });
    });

    it("accumulates multiple calls", () => {
      bridge.recordFromLmscript("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });
      bridge.recordFromLmscript("gpt-4", {
        promptTokens: 200,
        completionTokens: 100,
        totalTokens: 300,
      });

      const asUsage = tracker.currentTurnUsage();
      expect(asUsage.inputTokens).toBe(300);
      expect(asUsage.outputTokens).toBe(150);
      expect(costTracker.trackUsage).toHaveBeenCalledTimes(2);
    });
  });

  describe("recordFromAsAgent", () => {
    it("records usage in both trackers", () => {
      bridge.recordFromAsAgent(makeAsUsage(200, 100, 10, 5));

      // Check as-agent tracker
      const asUsage = tracker.currentTurnUsage();
      expect(asUsage.inputTokens).toBe(200);
      expect(asUsage.outputTokens).toBe(100);

      // Check lmscript cost tracker (converted format)
      expect(costTracker.trackUsage).toHaveBeenCalledWith("as-agent", {
        promptTokens: 200,
        completionTokens: 100,
        totalTokens: 315, // 200 + 100 + 10 + 5
      });
    });

    it("uses custom function name when provided", () => {
      bridge.recordFromAsAgent(makeAsUsage(50, 25), "wasm-call");

      expect(costTracker.trackUsage).toHaveBeenCalledWith("wasm-call", expect.any(Object));
    });
  });

  describe("endTurn", () => {
    it("advances the turn counter", () => {
      bridge.recordFromLmscript("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });
      bridge.endTurn();

      expect(tracker.turns()).toBe(1);
    });

    it("resets current turn usage but preserves cumulative", () => {
      bridge.recordFromLmscript("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });
      bridge.endTurn();

      // Current turn should be fresh
      const current = tracker.currentTurnUsage();
      expect(current.inputTokens).toBe(0);

      // Cumulative should retain data
      const cumulative = tracker.cumulativeUsage();
      expect(cumulative.inputTokens).toBe(100);
    });
  });

  describe("getCostSummary", () => {
    it("returns unified summary with both views", () => {
      bridge.recordFromLmscript("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });
      bridge.endTurn();

      const summary = bridge.getCostSummary(undefined, makeAsPricing());

      // as-agent view
      expect(summary.asAgent.usage.inputTokens).toBe(100);
      expect(summary.asAgent.usage.outputTokens).toBe(50);
      expect(summary.asAgent.turns).toBe(1);
      expect(summary.asAgent.costEstimate).toBeDefined();
      expect(summary.asAgent.costEstimate!.totalCostUsd).toBeGreaterThan(0);

      // lmscript view
      expect(summary.lmscript).toBeDefined();
      expect(summary.lmscript!.totalTokens).toBe(150);
    });

    it("returns undefined lmscript view when no cost tracker", () => {
      const bridgeNoCost = new UsageBridge(tracker);
      bridgeNoCost.recordFromLmscript("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });

      const summary = bridgeNoCost.getCostSummary();

      expect(summary.asAgent.usage.inputTokens).toBe(100);
      expect(summary.lmscript).toBeUndefined();
    });

    it("omits cost estimate when no as-agent pricing provided", () => {
      bridge.recordFromLmscript("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });

      const summary = bridge.getCostSummary();
      expect(summary.asAgent.costEstimate).toBeUndefined();
    });
  });

  describe("reset", () => {
    it("resets both trackers", () => {
      bridge.recordFromLmscript("gpt-4", {
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      });
      bridge.endTurn();

      bridge.reset();

      expect(tracker.turns()).toBe(0);
      expect(tracker.cumulativeUsage().inputTokens).toBe(0);
      expect(costTracker.reset).toHaveBeenCalled();
    });
  });

  describe("getAsTracker / getLmTracker", () => {
    it("returns the as-agent tracker", () => {
      expect(bridge.getAsTracker()).toBe(tracker);
    });

    it("returns the lmscript cost tracker", () => {
      expect(bridge.getLmTracker()).toBe(costTracker);
    });

    it("returns undefined lm tracker when not provided", () => {
      const bridgeNoCost = new UsageBridge(tracker);
      expect(bridgeNoCost.getLmTracker()).toBeUndefined();
    });
  });
});

// ── Roundtrip consistency ────────────────────────────────────────────

describe("Roundtrip conversion consistency", () => {
  it("lmscript -> as-agent -> lmscript preserves promptTokens and completionTokens", () => {
    const original = { promptTokens: 500, completionTokens: 250, totalTokens: 750 };
    const asFormat = lmscriptToAsTokenUsage(original);
    const backToLm = asTokenUsageToLmscript(asFormat);

    expect(backToLm.promptTokens).toBe(original.promptTokens);
    expect(backToLm.completionTokens).toBe(original.completionTokens);
    // Total will equal promptTokens + completionTokens (cache tokens are 0)
    expect(backToLm.totalTokens).toBe(750);
  });

  it("as-agent -> lmscript -> as-agent preserves input/output tokens", () => {
    const original = makeAsUsage(300, 150, 0, 0);
    const lmFormat = asTokenUsageToLmscript(original);
    const backToAs = lmscriptToAsTokenUsage(lmFormat);

    expect(backToAs.inputTokens).toBe(original.inputTokens);
    expect(backToAs.outputTokens).toBe(original.outputTokens);
  });

  it("cache tokens are lost in roundtrip (lmscript has no cache concept)", () => {
    const original = makeAsUsage(300, 150, 50, 25);
    const lmFormat = asTokenUsageToLmscript(original);
    const backToAs = lmscriptToAsTokenUsage(lmFormat);

    // Cache tokens are zeroed out on the way back
    expect(backToAs.cacheCreationInputTokens).toBe(0);
    expect(backToAs.cacheReadInputTokens).toBe(0);
    // But input/output tokens survive
    expect(backToAs.inputTokens).toBe(300);
    expect(backToAs.outputTokens).toBe(150);
  });
});
