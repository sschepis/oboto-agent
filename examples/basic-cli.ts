/**
 * Basic CLI Agent Example
 *
 * A minimal interactive CLI agent that demonstrates the dual-LLM
 * triage pattern with a simple tool set.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=sk-... npx tsx examples/basic-cli.ts
 *
 * Requires:
 *   - Ollama running locally with llama3:8b pulled
 *   - An Anthropic API key
 */

import * as readline from "readline";
import { ObotoAgent } from "../src/index.js";
import { OllamaProvider, AnthropicProvider } from "@sschepis/lmscript";
import {
  TreeBuilder,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";

// ── Build Tools ──────────────────────────────────────────────────────

const builder = new TreeBuilder();

builder.leaf("time", "Get the current date and time", {}, () =>
  new Date().toISOString()
);

builder.leaf(
  "calculate",
  "Evaluate a math expression",
  { expression: { type: "string" as const, description: "Math expression" } },
  (args) => {
    try {
      // Simple safe eval for basic math
      const expr = String(args.expression).replace(/[^0-9+\-*/().% ]/g, "");
      return String(Function(`"use strict"; return (${expr})`)());
    } catch {
      return "Error: Invalid expression";
    }
  }
);

builder.leaf(
  "echo",
  "Echo back a message",
  { message: { type: "string" as const, description: "Message to echo" } },
  (args) => `Echo: ${args.message}`
);

const { root } = builder.build();
const router = new Router(root, new SessionManager("cli-session"));

// ── Create Agent ─────────────────────────────────────────────────────

const agent = new ObotoAgent({
  localModel: new OllamaProvider({ baseUrl: "http://localhost:11434" }),
  remoteModel: new AnthropicProvider({
    apiKey: process.env.ANTHROPIC_API_KEY!,
  }),
  localModelName: "llama3:8b",
  remoteModelName: "claude-sonnet-4-20250514",
  router,
  systemPrompt:
    "You are a helpful assistant. You can tell the time, do math, and echo messages.",
});

// ── Wire Events ──────────────────────────────────────────────────────

agent.on("triage_result", (e) => {
  const { escalate, reasoning } = e.payload as any;
  console.log(
    `\n[triage] ${escalate ? "Escalating" : "Handling locally"}: ${reasoning}`
  );
});

agent.on("agent_thought", (e) => {
  const { text, model } = e.payload as any;
  console.log(`\n[${model}] ${text}`);
});

agent.on("tool_execution_start", (e) => {
  const { command, kwargs } = e.payload as any;
  console.log(`  > Tool: ${command}`, kwargs);
});

agent.on("tool_execution_complete", (e) => {
  const { result } = e.payload as any;
  console.log(`  < Result: ${result}`);
});

agent.on("error", (e) => {
  console.error(`\n[error] ${(e.payload as any).message}`);
});

// ── Interactive Loop ─────────────────────────────────────────────────

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: "\nyou> ",
});

console.log("oboto-agent CLI demo");
console.log("Type a message, or 'quit' to exit.\n");
rl.prompt();

rl.on("line", async (line) => {
  const input = line.trim();
  if (!input) {
    rl.prompt();
    return;
  }
  if (input === "quit" || input === "exit") {
    console.log("Goodbye!");
    rl.close();
    process.exit(0);
  }

  await agent.submitInput(input);
  rl.prompt();
});
