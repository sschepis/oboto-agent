/**
 * as-agent-features.ts — Deep integration of @sschepis/as-agent's advanced features.
 *
 * This module bridges the following as-agent subsystems into oboto-agent:
 *
 * 1. **Permissions** — PermissionPolicy + PermissionPrompter for tool authorization
 * 2. **Session Compaction** — CompactionConfig for auto-compacting long sessions
 * 3. **Hooks** — HookRunner for pre/post tool-use shell hooks
 * 4. **Slash Commands** — AgentRuntime's slash command registry
 * 5. **Usage Tracking** — UsageTracker bridge (detailed impl in Task 11)
 */

import type {
  PermissionPolicy,
  PermissionPrompter,
  PermissionOutcome,
  PermissionMode,
  CompactionConfig,
  CompactionResult,
  HookRunner,
  HookRunResult,
  SlashCommandSpec,
  AgentRuntime,
  Session,
  ConversationMessage,
  TokenUsage as AsTokenUsage,
  UsageTracker,
} from "@sschepis/as-agent";
import { MessageRole } from "@sschepis/as-agent";
import type { AgentEventBus } from "../event-bus.js";

// ── Permission Guard ──────────────────────────────────────────────────

/**
 * Wraps an as-agent PermissionPolicy to provide tool-call authorization
 * that integrates with the agent's event bus.
 *
 * Before each tool execution, `authorize()` is called. If the policy
 * denies the call, a `permission_denied` event is emitted and the tool
 * execution is skipped.
 *
 * ```ts
 * const guard = new PermissionGuard(policy, prompter, bus);
 * const outcome = guard.checkPermission("bash", '{"cmd":"rm -rf /"}');
 * if (outcome.kind === "deny") { ... }
 * ```
 */
export class PermissionGuard {
  constructor(
    private policy: PermissionPolicy,
    private prompter: PermissionPrompter | null,
    private bus?: AgentEventBus,
  ) {}

  /**
   * Check whether a tool call is authorized.
   * Emits `permission_denied` on the event bus if denied.
   */
  checkPermission(toolName: string, toolInput: string): PermissionOutcome {
    const outcome = this.policy.authorize(toolName, toolInput, this.prompter);

    if (outcome.kind === "deny") {
      this.bus?.emit("permission_denied", {
        toolName,
        toolInput,
        reason: outcome.reason ?? "denied by policy",
        activeMode: this.policy.activeMode,
        requiredMode: this.policy.requiredModeFor(toolName),
      });
    }

    return outcome;
  }

  /** Get the current active permission mode. */
  get activeMode(): PermissionMode {
    return this.policy.activeMode;
  }

  /** Get the required permission mode for a specific tool. */
  requiredModeFor(toolName: string): PermissionMode {
    return this.policy.requiredModeFor(toolName);
  }
}

// ── Session Compactor ─────────────────────────────────────────────────

/**
 * Manages automatic and manual session compaction using as-agent's
 * CompactionConfig. Compaction summarizes older messages to stay within
 * token limits while preserving recent context.
 *
 * ```ts
 * const compactor = new SessionCompactor(bus, {
 *   preserveRecentMessages: 10,
 *   maxEstimatedTokens: 100_000,
 * });
 *
 * // Check and compact if needed
 * const result = compactor.compactIfNeeded(session, estimatedTokens);
 * ```
 */
export class SessionCompactor {
  private config: CompactionConfig;
  private bus?: AgentEventBus;
  private compactionCount = 0;
  private totalRemovedMessages = 0;

  constructor(bus: AgentEventBus | undefined, config: CompactionConfig) {
    this.bus = bus;
    this.config = config;
  }

  /**
   * Check if the session needs compaction and perform it if so.
   * Uses a simple heuristic: ~4 chars per token for estimation.
   *
   * Since the actual compaction logic lives in the Wasm runtime
   * (ConversationRuntime.compact()), this method provides a JS-side
   * implementation that creates a summary of older messages and
   * preserves recent ones.
   *
   * Returns null if compaction is not needed.
   */
  compactIfNeeded(
    session: Session,
    estimatedTokens?: number,
  ): CompactionResult | null {
    const tokenEstimate = estimatedTokens ?? this.estimateTokens(session);

    if (tokenEstimate <= this.config.maxEstimatedTokens) {
      return null;
    }

    return this.compact(session);
  }

  /**
   * Force compaction of the session regardless of token count.
   */
  compact(session: Session): CompactionResult {
    const preserve = this.config.preserveRecentMessages;
    const totalMessages = session.messages.length;

    if (totalMessages <= preserve) {
      // Nothing to compact
      return {
        summary: "",
        formattedSummary: "",
        compactedSession: session,
        removedMessageCount: 0,
      };
    }

    // Separate messages into "to summarize" and "to preserve"
    const toSummarize = session.messages.slice(0, totalMessages - preserve);
    const toPreserve = session.messages.slice(totalMessages - preserve);

    // Build a summary from the messages being removed
    const summaryParts: string[] = [];
    for (const msg of toSummarize) {
      const role = roleToString(msg.role);
      const text = blocksToText(msg.blocks);
      if (text.trim()) {
        summaryParts.push(`[${role}]: ${text}`);
      }
    }

    const summary = summaryParts.join("\n\n");
    const formattedSummary = `[Session Compaction Summary — ${toSummarize.length} messages summarized]\n\n${summary}`;

    // Build the compacted session with a summary message + preserved messages
    const summaryMessage: ConversationMessage = {
      role: MessageRole.System,
      blocks: [{
        kind: "text",
        text: `[Previous conversation summary — ${toSummarize.length} messages compacted]\n\n${
          summary.length > 4000 ? summary.slice(0, 4000) + "\n\n[... summary truncated]" : summary
        }`,
      }],
    };

    const compactedSession: Session = {
      version: session.version,
      messages: [summaryMessage, ...toPreserve],
    };

    const result: CompactionResult = {
      summary,
      formattedSummary,
      compactedSession,
      removedMessageCount: toSummarize.length,
    };

    this.compactionCount++;
    this.totalRemovedMessages += toSummarize.length;

    this.bus?.emit("session_compacted", {
      removedMessageCount: toSummarize.length,
      preservedMessageCount: toPreserve.length + 1, // +1 for summary msg
      estimatedTokensBefore: this.estimateTokens(session),
      estimatedTokensAfter: this.estimateTokens(compactedSession),
      compactionIndex: this.compactionCount,
    });

    return result;
  }

  /** Get compaction statistics. */
  get stats() {
    return {
      compactionCount: this.compactionCount,
      totalRemovedMessages: this.totalRemovedMessages,
    };
  }

  /** Update the compaction config at runtime. */
  updateConfig(config: Partial<CompactionConfig>): void {
    if (config.preserveRecentMessages !== undefined) {
      this.config.preserveRecentMessages = config.preserveRecentMessages;
    }
    if (config.maxEstimatedTokens !== undefined) {
      this.config.maxEstimatedTokens = config.maxEstimatedTokens;
    }
  }

  /** Estimate token count for a session (~4 chars per token). */
  private estimateTokens(session: Session): number {
    let charCount = 0;
    for (const msg of session.messages) {
      for (const block of msg.blocks) {
        if (block.kind === "text") {
          charCount += block.text.length;
        } else if (block.kind === "tool_use") {
          charCount += block.name.length + block.input.length;
        } else if (block.kind === "tool_result") {
          charCount += block.output.length;
        }
      }
    }
    return Math.ceil(charCount / 4);
  }
}

// ── Hook Integration ──────────────────────────────────────────────────

/**
 * Wraps an as-agent HookRunner and integrates it with the agent's
 * event bus. Hooks are shell commands that run before and after
 * tool executions, allowing external validation/transformation.
 *
 * ```ts
 * const hooks = new HookIntegration(hookRunner, bus);
 * const pre = hooks.runPreToolUse("bash", '{"cmd":"ls"}');
 * if (pre.denied) { /* skip tool call *\/ }
 *
 * // After tool executes:
 * const post = hooks.runPostToolUse("bash", '{"cmd":"ls"}', "file1\nfile2", false);
 * ```
 */
export class HookIntegration {
  constructor(
    private runner: HookRunner,
    private bus?: AgentEventBus,
  ) {}

  /**
   * Run pre-tool-use hooks. If any hook denies the call,
   * the tool execution should be skipped.
   */
  runPreToolUse(toolName: string, toolInput: string): HookRunResult {
    const result = this.runner.runPreToolUse(toolName, toolInput);

    if (result.denied) {
      this.bus?.emit("hook_denied", {
        phase: "pre",
        toolName,
        toolInput,
        messages: result.messages,
      });
    } else if (result.messages.length > 0) {
      this.bus?.emit("hook_message", {
        phase: "pre",
        toolName,
        messages: result.messages,
      });
    }

    return result;
  }

  /**
   * Run post-tool-use hooks. These can log, transform, or audit
   * tool results but cannot retroactively deny execution.
   */
  runPostToolUse(
    toolName: string,
    toolInput: string,
    toolOutput: string,
    isError: boolean,
  ): HookRunResult {
    const result = this.runner.runPostToolUse(toolName, toolInput, toolOutput, isError);

    if (result.messages.length > 0) {
      this.bus?.emit("hook_message", {
        phase: "post",
        toolName,
        messages: result.messages,
      });
    }

    return result;
  }
}

// ── Slash Command Registry ────────────────────────────────────────────

/**
 * Bridges the as-agent Wasm runtime's slash command system with
 * oboto-agent. The Wasm runtime provides 21 built-in slash commands
 * (e.g. /help, /compact, /clear, /model, etc.).
 *
 * This class provides:
 * - Access to command specs and help text from the Wasm runtime
 * - An extensible registry for adding custom slash commands
 * - Parsing and dispatching of slash command input
 *
 * ```ts
 * const registry = new SlashCommandRegistry(agentRuntime);
 * const specs = registry.getCommandSpecs();
 * const help = registry.getHelpText();
 *
 * // Add custom commands
 * registry.registerCommand({
 *   name: "status",
 *   summary: "Show agent status",
 *   argumentHint: "",
 *   resumeSupported: false,
 * }, async (args) => "Agent is running");
 * ```
 */
export class SlashCommandRegistry {
  private customCommands = new Map<string, {
    spec: SlashCommandSpec;
    handler: (args: string) => string | Promise<string>;
  }>();
  private wasmRuntime?: AgentRuntime;

  constructor(wasmRuntime?: AgentRuntime) {
    this.wasmRuntime = wasmRuntime;
  }

  /**
   * Get all slash command specs (built-in from Wasm + custom).
   */
  getCommandSpecs(): SlashCommandSpec[] {
    const builtIn = this.wasmRuntime
      ? (this.wasmRuntime.slashCommandSpecs() as SlashCommandSpec[])
      : [];

    const custom = Array.from(this.customCommands.values()).map(c => c.spec);

    return [...builtIn, ...custom];
  }

  /**
   * Get the full help text for all commands.
   * Uses the Wasm runtime's formatted help if available.
   */
  getHelpText(): string {
    const parts: string[] = [];

    if (this.wasmRuntime) {
      parts.push(this.wasmRuntime.renderSlashCommandHelp());
    }

    if (this.customCommands.size > 0) {
      parts.push("\n## Custom Commands\n");
      for (const [name, { spec }] of this.customCommands) {
        const hint = spec.argumentHint ? ` ${spec.argumentHint}` : "";
        parts.push(`/${name}${hint} — ${spec.summary}`);
      }
    }

    return parts.join("\n");
  }

  /**
   * Register a custom slash command.
   */
  registerCommand(
    spec: SlashCommandSpec,
    handler: (args: string) => string | Promise<string>,
  ): void {
    this.customCommands.set(spec.name, { spec, handler });
  }

  /**
   * Unregister a custom slash command.
   */
  unregisterCommand(name: string): boolean {
    return this.customCommands.delete(name);
  }

  /**
   * Parse a user input string to check if it's a slash command.
   * Returns the command name and arguments, or null if not a command.
   */
  parseCommand(input: string): { name: string; args: string } | null {
    const trimmed = input.trim();
    if (!trimmed.startsWith("/")) return null;

    const spaceIdx = trimmed.indexOf(" ");
    const name = spaceIdx === -1
      ? trimmed.slice(1)
      : trimmed.slice(1, spaceIdx);
    const args = spaceIdx === -1
      ? ""
      : trimmed.slice(spaceIdx + 1).trim();

    return { name, args };
  }

  /**
   * Execute a custom slash command by name.
   * Returns the command output, or null if the command is not found
   * in the custom registry (it may be a built-in Wasm command).
   */
  async executeCustomCommand(name: string, args: string): Promise<string | null> {
    const entry = this.customCommands.get(name);
    if (!entry) return null;
    return entry.handler(args);
  }

  /**
   * Check if a command name exists (either built-in or custom).
   */
  hasCommand(name: string): boolean {
    if (this.customCommands.has(name)) return true;
    const specs = this.getCommandSpecs();
    return specs.some(s => s.name === name);
  }

  /**
   * Get only commands that support resume (useful for session restoration).
   */
  getResumeSupportedCommands(): SlashCommandSpec[] {
    const builtIn = this.wasmRuntime
      ? (this.wasmRuntime.resumeSupportedSlashCommands() as SlashCommandSpec[])
      : [];

    const custom = Array.from(this.customCommands.values())
      .filter(c => c.spec.resumeSupported)
      .map(c => c.spec);

    return [...builtIn, ...custom];
  }
}

// ── Usage Tracker Adapter ─────────────────────────────────────────────

/**
 * A JS-side implementation of the as-agent UsageTracker interface.
 * This adapter accumulates token usage across turns, matching the
 * as-agent contract (inputTokens, outputTokens, cache tokens).
 *
 * The bridge to lmscript's CostTracker (which uses different field names)
 * is handled in Task 11.
 */
export class AgentUsageTracker implements UsageTracker {
  private turnUsages: AsTokenUsage[] = [];
  private currentTurn: AsTokenUsage = {
    inputTokens: 0,
    outputTokens: 0,
    cacheCreationInputTokens: 0,
    cacheReadInputTokens: 0,
  };

  record(usage: AsTokenUsage): void {
    this.currentTurn = {
      inputTokens: this.currentTurn.inputTokens + usage.inputTokens,
      outputTokens: this.currentTurn.outputTokens + usage.outputTokens,
      cacheCreationInputTokens: this.currentTurn.cacheCreationInputTokens + usage.cacheCreationInputTokens,
      cacheReadInputTokens: this.currentTurn.cacheReadInputTokens + usage.cacheReadInputTokens,
    };
  }

  currentTurnUsage(): AsTokenUsage {
    return { ...this.currentTurn };
  }

  cumulativeUsage(): AsTokenUsage {
    const cumulative: AsTokenUsage = {
      inputTokens: 0,
      outputTokens: 0,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 0,
    };

    for (const usage of this.turnUsages) {
      cumulative.inputTokens += usage.inputTokens;
      cumulative.outputTokens += usage.outputTokens;
      cumulative.cacheCreationInputTokens += usage.cacheCreationInputTokens;
      cumulative.cacheReadInputTokens += usage.cacheReadInputTokens;
    }

    // Add current turn
    cumulative.inputTokens += this.currentTurn.inputTokens;
    cumulative.outputTokens += this.currentTurn.outputTokens;
    cumulative.cacheCreationInputTokens += this.currentTurn.cacheCreationInputTokens;
    cumulative.cacheReadInputTokens += this.currentTurn.cacheReadInputTokens;

    return cumulative;
  }

  turns(): number {
    return this.turnUsages.length + (this.hasTurnActivity() ? 1 : 0);
  }

  /**
   * Finalize the current turn and start a new one.
   * Call this at the end of each agent turn.
   */
  endTurn(): void {
    if (this.hasTurnActivity()) {
      this.turnUsages.push({ ...this.currentTurn });
      this.currentTurn = {
        inputTokens: 0,
        outputTokens: 0,
        cacheCreationInputTokens: 0,
        cacheReadInputTokens: 0,
      };
    }
  }

  /** Reset all usage data. */
  reset(): void {
    this.turnUsages = [];
    this.currentTurn = {
      inputTokens: 0,
      outputTokens: 0,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 0,
    };
  }

  private hasTurnActivity(): boolean {
    return (
      this.currentTurn.inputTokens > 0 ||
      this.currentTurn.outputTokens > 0
    );
  }
}

// ── Helpers ───────────────────────────────────────────────────────────

function roleToString(role: MessageRole): string {
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
