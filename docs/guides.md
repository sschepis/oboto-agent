# Guides

## Getting Started

### Installation

```bash
npm install @sschepis/oboto-agent
# Peer dependencies (install the ones you need):
npm install @sschepis/lmscript @sschepis/swiss-army-tool @sschepis/as-agent
```

### Basic Setup

```typescript
import { ObotoAgent } from "@sschepis/oboto-agent";
import { OllamaProvider, AnthropicProvider } from "@sschepis/lmscript";
import { TreeBuilder, Router, SessionManager } from "@sschepis/swiss-army-tool";

// 1. Build a tool tree
const builder = new TreeBuilder();
builder.leaf("help", "Show help", {}, () => "Available commands: help, greet");
builder.leaf("greet", "Say hello", { name: "string" }, (args) => `Hello, ${args.name}!`);
const { root } = builder.build();

// 2. Create a router
const router = new Router(root, new SessionManager("session-1"));

// 3. Create the agent
const agent = new ObotoAgent({
  localModel: new OllamaProvider({ baseUrl: "http://localhost:11434" }),
  remoteModel: new AnthropicProvider({ apiKey: process.env.ANTHROPIC_API_KEY! }),
  localModelName: "llama3:8b",
  remoteModelName: "claude-sonnet-4-20250514",
  router,
});

// 4. Listen to events
agent.on("agent_thought", (e) => process.stdout.write(e.payload.text));
agent.on("error", (e) => console.error("Error:", e.payload.message));

// 5. Submit input
await agent.submitInput("Hello, can you greet Alice for me?");
```

---

## Building a CLI Agent

```typescript
import * as readline from "readline";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Setup agent (as above)...

agent.on("agent_thought", (e) => {
  console.log(`\n[${e.payload.model}] ${e.payload.text}`);
});

agent.on("tool_execution_start", (e) => {
  console.log(`  > Running: ${e.payload.command}`);
});

agent.on("tool_execution_complete", (e) => {
  console.log(`  < Result: ${e.payload.result}`);
});

agent.on("turn_complete", () => {
  rl.prompt();
});

rl.on("line", async (line) => {
  const input = line.trim();
  if (input === "quit") {
    rl.close();
    return;
  }
  await agent.submitInput(input);
});

rl.prompt();
```

---

## Resuming a Session

Sessions are plain JSON objects. Serialize them to disk and restore later:

```typescript
import { writeFileSync, readFileSync, existsSync } from "fs";

const SESSION_FILE = "./session.json";

// Load existing session
const session = existsSync(SESSION_FILE)
  ? JSON.parse(readFileSync(SESSION_FILE, "utf-8"))
  : undefined;

const agent = new ObotoAgent({
  // ...providers and router...
  session,
});

// Save session on state changes
agent.on("state_updated", () => {
  writeFileSync(SESSION_FILE, JSON.stringify(agent.getSession(), null, 2));
});
```

---

## Custom Tools

Build a rich tool tree with swiss-army-tool's `TreeBuilder`:

```typescript
import { TreeBuilder } from "@sschepis/swiss-army-tool";
import { execSync } from "child_process";
import { readFileSync, writeFileSync, readdirSync } from "fs";

const builder = new TreeBuilder();

builder.branch("fs", "File system operations", (b) => {
  b.leaf("read", "Read a file", { path: "string" }, (args) =>
    readFileSync(String(args.path), "utf-8")
  );
  b.leaf("write", "Write a file", { path: "string", content: "string" }, (args) => {
    writeFileSync(String(args.path), String(args.content));
    return `Written to ${args.path}`;
  });
  b.leaf("ls", "List directory", { path: "string" }, (args) =>
    readdirSync(String(args.path)).join("\n")
  );
});

builder.branch("shell", "Shell commands", (b) => {
  b.leaf("exec", "Execute a command", { cmd: "string" }, (args) =>
    execSync(String(args.cmd), { encoding: "utf-8", timeout: 30000 })
  );
});

const { root } = builder.build();
```

---

## Handling Interruptions

When a user sends new input while the agent is mid-execution:

```typescript
agent.on("interruption", (e) => {
  console.log("Interrupted!", e.payload.newDirectives);
});

// First request starts processing
agent.submitInput("Analyze all files in the project");

// User changes their mind — this triggers interrupt()
setTimeout(() => {
  agent.submitInput("Actually, just look at package.json");
}, 500);
```

---

## Using with Different Providers

oboto-agent works with any lmscript provider:

```typescript
import {
  OllamaProvider,
  LMStudioProvider,
  AnthropicProvider,
  OpenAIProvider,
  GeminiProvider,
  OpenRouterProvider,
  DeepSeekProvider,
} from "@sschepis/lmscript";

// Local options
const local = new OllamaProvider({ baseUrl: "http://localhost:11434" });
// const local = new LMStudioProvider({ baseUrl: "http://localhost:1234" });

// Remote options
const remote = new AnthropicProvider({ apiKey: "..." });
// const remote = new OpenAIProvider({ apiKey: "..." });
// const remote = new GeminiProvider({ apiKey: "..." });
// const remote = new OpenRouterProvider({ apiKey: "..." });
// const remote = new DeepSeekProvider({ apiKey: "..." });
```

---

## Custom System Prompts

```typescript
const agent = new ObotoAgent({
  // ...
  systemPrompt: `You are a DevOps assistant specialized in Kubernetes.
You help users manage their clusters, debug deployments, and write manifests.
Always explain what you're doing before executing commands.
Prefer kubectl over direct API calls.`,
});
```
