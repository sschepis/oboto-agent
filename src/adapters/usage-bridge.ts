/**
 * usage-bridge.ts — Bidirectional bridge between as-agent's UsageTracker
 * and lmscript's CostTracker.
 *
 * Field mapping:
 *   as-agent                      lmscript
 *   ─────────                     ────────
 *   inputTokens              →    promptTokens
 *   outputTokens             →    completionTokens
 *   cacheCreationInputTokens →    (no equivalent, tracked separately)
 *   cacheReadInputTokens     →    (no equivalent, tracked separately)
 *   (sum of all)             →    totalTokens
 *
 * The bridge records every usage event into both systems, ensuring
 * unified cost accounting regardless of which subsystem reports usage.
 */

import type {
  TokenUsage as AsTokenUsage,
  UsageTracker,
  ModelPricing as AsModelPricing,
  UsageCostEstimate,
} from "@sschepis/as-agent";
import type {
  CostTracker,
  ModelPricing as LmModelPricing,
} from "@sschepis/lmscript";
import { AgentUsageTracker } from "./as-agent-features.js";

// ── Type converters ──────────────────────────────────────────────────

/**
 * Convert as-agent TokenUsage to lmscript usage format.
 */
export function asTokenUsageToLmscript(usage: AsTokenUsage): {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
} {
  const promptTokens = usage.inputTokens;
  const completionTokens = usage.outputTokens;
  // Total includes all token types (input + output + cache tokens)
  const totalTokens =
    usage.inputTokens +
    usage.outputTokens +
    usage.cacheCreationInputTokens +
    usage.cacheReadInputTokens;

  return { promptTokens, completionTokens, totalTokens };
}

/**
 * Convert lmscript usage format to as-agent TokenUsage.
 * Cache tokens default to 0 since lmscript doesn't track them separately.
 */
export function lmscriptToAsTokenUsage(usage: {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}): AsTokenUsage {
  return {
    inputTokens: usage.promptTokens,
    outputTokens: usage.completionTokens,
    cacheCreationInputTokens: 0,
    cacheReadInputTokens: 0,
  };
}

// ── Cost estimation ──────────────────────────────────────────────────

/**
 * Estimate costs using as-agent's ModelPricing (per-million-token pricing).
 */
export function estimateCostFromAsAgent(
  usage: AsTokenUsage,
  pricing: AsModelPricing,
): UsageCostEstimate {
  const inputCostUsd = (usage.inputTokens / 1_000_000) * pricing.inputCostPerMillion;
  const outputCostUsd = (usage.outputTokens / 1_000_000) * pricing.outputCostPerMillion;
  const cacheCreationCostUsd = (usage.cacheCreationInputTokens / 1_000_000) * pricing.cacheCreationCostPerMillion;
  const cacheReadCostUsd = (usage.cacheReadInputTokens / 1_000_000) * pricing.cacheReadCostPerMillion;
  const totalCostUsd = inputCostUsd + outputCostUsd + cacheCreationCostUsd + cacheReadCostUsd;

  return {
    inputCostUsd,
    outputCostUsd,
    cacheCreationCostUsd,
    cacheReadCostUsd,
    totalCostUsd,
  };
}

// ── Usage Bridge ─────────────────────────────────────────────────────

/**
 * Bidirectional bridge that keeps an as-agent UsageTracker and an
 * lmscript CostTracker in sync.
 *
 * When usage is reported from either side, the bridge records it in
 * both tracking systems.
 *
 * ```ts
 * const bridge = new UsageBridge(usageTracker, costTracker);
 *
 * // Record from LLM response (lmscript format)
 * bridge.recordFromLmscript("gpt-4", { promptTokens: 100, completionTokens: 50, totalTokens: 150 });
 *
 * // Record from API response (as-agent format)
 * bridge.recordFromAsAgent({ inputTokens: 100, outputTokens: 50, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 });
 *
 * // Get unified cost summary
 * const summary = bridge.getCostSummary(lmModelPricing, asModelPricing);
 * ```
 */
export class UsageBridge {
  constructor(
    private asTracker: AgentUsageTracker,
    private lmTracker?: CostTracker,
  ) {}

  /**
   * Record usage from an lmscript-format source (e.g. from the streaming path
   * or AgentLoop result).
   *
   * Converts to as-agent format and records in both trackers.
   */
  recordFromLmscript(
    functionName: string,
    usage: { promptTokens: number; completionTokens: number; totalTokens: number },
  ): void {
    // Record in lmscript CostTracker
    this.lmTracker?.trackUsage(functionName, usage);

    // Convert and record in as-agent tracker
    this.asTracker.record(lmscriptToAsTokenUsage(usage));
  }

  /**
   * Record usage from an as-agent-format source (e.g. from ConversationRuntime
   * or the Wasm runtime).
   *
   * Converts to lmscript format and records in both trackers.
   */
  recordFromAsAgent(
    usage: AsTokenUsage,
    functionName?: string,
  ): void {
    // Record in as-agent tracker
    this.asTracker.record(usage);

    // Convert and record in lmscript CostTracker
    if (this.lmTracker) {
      this.lmTracker.trackUsage(
        functionName ?? "as-agent",
        asTokenUsageToLmscript(usage),
      );
    }
  }

  /**
   * End the current turn in the as-agent tracker.
   */
  endTurn(): void {
    this.asTracker.endTurn();
  }

  /**
   * Get unified cost summary combining both tracking systems.
   */
  getCostSummary(
    lmPricing?: LmModelPricing,
    asPricing?: AsModelPricing,
  ): UnifiedCostSummary {
    const asUsage = this.asTracker.cumulativeUsage();
    const asTurns = this.asTracker.turns();

    // as-agent side cost estimation
    let asCostEstimate: UsageCostEstimate | undefined;
    if (asPricing) {
      asCostEstimate = estimateCostFromAsAgent(asUsage, asPricing);
    }

    // lmscript side totals
    let lmTotalTokens: number | undefined;
    let lmTotalCost: number | undefined;
    let lmByFunction: Record<string, { calls: number; totalTokens: number; promptTokens: number; completionTokens: number }> | undefined;

    if (this.lmTracker) {
      lmTotalTokens = this.lmTracker.getTotalTokens();
      lmTotalCost = this.lmTracker.getTotalCost(lmPricing);
      const usageMap = this.lmTracker.getUsageByFunction();
      lmByFunction = {};
      for (const [fnName, entry] of usageMap) {
        lmByFunction[fnName] = entry;
      }
    }

    return {
      // as-agent view
      asAgent: {
        usage: asUsage,
        turns: asTurns,
        costEstimate: asCostEstimate,
      },
      // lmscript view
      lmscript: this.lmTracker ? {
        totalTokens: lmTotalTokens!,
        totalCost: lmTotalCost!,
        byFunction: lmByFunction!,
      } : undefined,
    };
  }

  /**
   * Reset both tracking systems.
   */
  reset(): void {
    this.asTracker.reset();
    this.lmTracker?.reset();
  }

  /** Get the as-agent usage tracker. */
  getAsTracker(): AgentUsageTracker {
    return this.asTracker;
  }

  /** Get the lmscript cost tracker (if available). */
  getLmTracker(): CostTracker | undefined {
    return this.lmTracker;
  }
}

// ── Unified summary type ─────────────────────────────────────────────

export interface UnifiedCostSummary {
  /** as-agent's view: per-turn token counts and optional cost estimate. */
  asAgent: {
    usage: AsTokenUsage;
    turns: number;
    costEstimate?: UsageCostEstimate;
  };
  /** lmscript's view: total tokens, cost, and per-function breakdown. */
  lmscript?: {
    totalTokens: number;
    totalCost: number;
    byFunction: Record<string, {
      calls: number;
      totalTokens: number;
      promptTokens: number;
      completionTokens: number;
    }>;
  };
}
