/**
 * Event Monitoring Example
 *
 * Demonstrates how to listen to all agent events for logging,
 * debugging, or building custom UIs.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=sk-... npx tsx examples/event-monitoring.ts
 */

import { ObotoAgent, type AgentEventType } from "../src/index.js";
import { OllamaProvider, AnthropicProvider } from "@sschepis/lmscript";
import {
  TreeBuilder,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";

// ── Minimal Tool Setup ───────────────────────────────────────────────

const builder = new TreeBuilder();
builder.leaf("ping", "Respond with pong", {}, () => "pong");
const { root } = builder.build();
const router = new Router(root, new SessionManager("monitor"));

// ── Create Agent ─────────────────────────────────────────────────────

const agent = new ObotoAgent({
  localModel: new OllamaProvider({ baseUrl: "http://localhost:11434" }),
  remoteModel: new AnthropicProvider({
    apiKey: process.env.ANTHROPIC_API_KEY!,
  }),
  localModelName: "llama3:8b",
  remoteModelName: "claude-sonnet-4-20250514",
  router,
});

// ── Monitor All Events ───────────────────────────────────────────────

const ALL_EVENTS: AgentEventType[] = [
  "user_input",
  "agent_thought",
  "triage_result",
  "tool_execution_start",
  "tool_execution_complete",
  "state_updated",
  "interruption",
  "error",
  "turn_complete",
];

const eventLog: Array<{ type: string; payload: unknown; time: string }> = [];

for (const eventType of ALL_EVENTS) {
  agent.on(eventType, (event) => {
    const entry = {
      type: event.type,
      payload: event.payload,
      time: new Date(event.timestamp).toISOString(),
    };
    eventLog.push(entry);

    // Pretty-print each event as it happens
    const color = getColor(event.type);
    console.log(
      `${color}[${entry.time}] ${event.type}${RESET}`,
      JSON.stringify(event.payload, null, 2)
    );
  });
}

// ANSI colors for terminal output
const RESET = "\x1b[0m";
function getColor(type: AgentEventType): string {
  const colors: Partial<Record<AgentEventType, string>> = {
    user_input: "\x1b[36m",       // cyan
    agent_thought: "\x1b[32m",    // green
    triage_result: "\x1b[33m",    // yellow
    tool_execution_start: "\x1b[34m",  // blue
    tool_execution_complete: "\x1b[34m",
    state_updated: "\x1b[90m",    // gray
    interruption: "\x1b[35m",     // magenta
    error: "\x1b[31m",            // red
    turn_complete: "\x1b[32m",    // green
  };
  return colors[type] ?? RESET;
}

// ── Run ──────────────────────────────────────────────────────────────

console.log("Event monitoring example\n");
console.log("Sending: 'Hello, what time is it?'\n");

await agent.submitInput("Hello, what time is it?");

console.log(`\n\n=== Event Summary ===`);
console.log(`Total events: ${eventLog.length}`);
const typeCounts = eventLog.reduce(
  (acc, e) => {
    acc[e.type] = (acc[e.type] ?? 0) + 1;
    return acc;
  },
  {} as Record<string, number>
);
for (const [type, count] of Object.entries(typeCounts)) {
  console.log(`  ${type}: ${count}`);
}
