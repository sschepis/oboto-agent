/**
 * Session Persistence Example
 *
 * Demonstrates how to save and restore agent sessions across
 * process restarts using simple JSON file persistence.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=sk-... npx tsx examples/session-persistence.ts
 *
 * Run it multiple times — the conversation carries over!
 */

import { writeFileSync, readFileSync, existsSync } from "fs";
import { ObotoAgent, createEmptySession } from "../src/index.js";
import { OllamaProvider, AnthropicProvider } from "@sschepis/lmscript";
import {
  TreeBuilder,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";
import type { Session } from "@sschepis/as-agent";

const SESSION_FILE = "./agent-session.json";

// ── Load or Create Session ───────────────────────────────────────────

function loadSession(): Session | undefined {
  if (existsSync(SESSION_FILE)) {
    console.log(`Resuming session from ${SESSION_FILE}`);
    const data = readFileSync(SESSION_FILE, "utf-8");
    return JSON.parse(data) as Session;
  }
  console.log("Starting fresh session");
  return undefined;
}

function saveSession(session: Session): void {
  writeFileSync(SESSION_FILE, JSON.stringify(session, null, 2));
  console.log(`Session saved (${session.messages.length} messages)`);
}

// ── Setup ────────────────────────────────────────────────────────────

const builder = new TreeBuilder();
builder.leaf("time", "Get current time", {}, () => new Date().toISOString());
builder.leaf("remember", "Store a note", { note: "string" as any }, (args) => {
  return `Noted: ${args.note}`;
});
const { root } = builder.build();
const router = new Router(root, new SessionManager("persistent"));

const session = loadSession();

const agent = new ObotoAgent({
  localModel: new OllamaProvider({ baseUrl: "http://localhost:11434" }),
  remoteModel: new AnthropicProvider({
    apiKey: process.env.ANTHROPIC_API_KEY!,
  }),
  localModelName: "llama3:8b",
  remoteModelName: "claude-sonnet-4-20250514",
  router,
  session,
  systemPrompt:
    "You are an assistant with memory. Recall previous conversations when relevant.",
});

// Auto-save on every state update
agent.on("state_updated", () => {
  saveSession(agent.getSession());
});

agent.on("agent_thought", (e) => {
  console.log(`\nAgent: ${(e.payload as any).text}`);
});

agent.on("error", (e) => {
  console.error(`Error: ${(e.payload as any).message}`);
});

// ── Run ──────────────────────────────────────────────────────────────

const query = process.argv[2] ?? "What have we talked about before?";
console.log(`\nYou: ${query}\n`);
await agent.submitInput(query);
console.log("\nDone. Run again with a different message to continue the conversation.");
console.log(`  npx tsx examples/session-persistence.ts "Your message here"`);
