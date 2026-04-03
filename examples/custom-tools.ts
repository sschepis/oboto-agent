/**
 * Custom Tools Example
 *
 * Demonstrates how to build a rich tool tree with swiss-army-tool
 * and wire it into oboto-agent. This example creates a file system
 * and shell tool set.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=sk-... npx tsx examples/custom-tools.ts
 */

import { ObotoAgent } from "../src/index.js";
import { OllamaProvider, AnthropicProvider } from "@sschepis/lmscript";
import {
  TreeBuilder,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";
import { readFileSync, readdirSync, statSync, existsSync } from "fs";
import { resolve } from "path";

// ── Build a File System Tool Tree ────────────────────────────────────

const builder = new TreeBuilder();

builder.branch("fs", "File system operations", (b) => {
  b.leaf(
    "read",
    "Read a file's contents",
    { path: { type: "string" as const, description: "File path to read" } },
    (args) => {
      const p = resolve(String(args.path));
      if (!existsSync(p)) return `Error: File not found: ${p}`;
      return readFileSync(p, "utf-8");
    }
  );

  b.leaf(
    "ls",
    "List directory contents",
    { path: { type: "string" as const, description: "Directory path" } },
    (args) => {
      const p = resolve(String(args.path));
      if (!existsSync(p)) return `Error: Directory not found: ${p}`;
      const entries = readdirSync(p);
      return entries
        .map((name) => {
          const stat = statSync(resolve(p, name));
          return `${stat.isDirectory() ? "d" : "-"} ${name}`;
        })
        .join("\n");
    }
  );

  b.leaf(
    "stat",
    "Get file info",
    { path: { type: "string" as const, description: "File path" } },
    (args) => {
      const p = resolve(String(args.path));
      if (!existsSync(p)) return `Error: Not found: ${p}`;
      const stat = statSync(p);
      return [
        `Path: ${p}`,
        `Type: ${stat.isDirectory() ? "directory" : "file"}`,
        `Size: ${stat.size} bytes`,
        `Modified: ${stat.mtime.toISOString()}`,
      ].join("\n");
    }
  );
});

builder.branch("info", "System information", (b) => {
  b.leaf("time", "Current date and time", {}, () => new Date().toISOString());
  b.leaf("cwd", "Current working directory", {}, () => process.cwd());
  b.leaf("env", "List environment variable names", {}, () =>
    Object.keys(process.env).sort().join("\n")
  );
});

const { root } = builder.build();
const router = new Router(root, new SessionManager("tools-demo"));

// ── Create Agent ─────────────────────────────────────────────────────

const agent = new ObotoAgent({
  localModel: new OllamaProvider({ baseUrl: "http://localhost:11434" }),
  remoteModel: new AnthropicProvider({
    apiKey: process.env.ANTHROPIC_API_KEY!,
  }),
  localModelName: "llama3:8b",
  remoteModelName: "claude-sonnet-4-20250514",
  router,
  systemPrompt: `You are a file system assistant. You can read files, list directories,
and provide system information. Always use the available tools to answer
questions about the file system. Describe what you find clearly.`,
});

// ── Run a Single Query ───────────────────────────────────────────────

agent.on("agent_thought", (e) => {
  console.log(`\n${(e.payload as any).text}`);
});

agent.on("tool_execution_start", (e) => {
  const { command, kwargs } = e.payload as any;
  console.log(`\n  [tool] ${command}`, JSON.stringify(kwargs));
});

agent.on("turn_complete", (e) => {
  const p = e.payload as any;
  console.log(
    `\n--- Turn complete (model: ${p.model}, escalated: ${p.escalated}) ---`
  );
});

agent.on("error", (e) => {
  console.error(`Error: ${(e.payload as any).message}`);
});

// Run
console.log("Custom tools example — querying the file system\n");
await agent.submitInput(
  "List the files in the current directory and read the package.json file. Summarize what this project is."
);
