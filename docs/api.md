# API Reference

## ObotoAgent

The main orchestrator class.

### Constructor

```typescript
new ObotoAgent(config: ObotoAgentConfig)
```

#### ObotoAgentConfig

| Property | Type | Required | Default | Description |
|---|---|---|---|---|
| `localModel` | `LLMProvider` | Yes | — | Small, fast model for triage and summarization |
| `remoteModel` | `LLMProvider` | Yes | — | Powerful model for complex tasks |
| `localModelName` | `string` | Yes | — | Model identifier for the local provider |
| `remoteModelName` | `string` | Yes | — | Model identifier for the remote provider |
| `router` | `Router` | Yes | — | Pre-built swiss-army-tool Router |
| `session` | `Session` | No | Empty session | Existing session to resume |
| `maxContextTokens` | `number` | No | `8192` | Token budget for context window |
| `maxIterations` | `number` | No | `10` | Max LLM iterations per turn |
| `systemPrompt` | `string` | No | Generic assistant prompt | System prompt for all LLM calls |

### Methods

#### `submitInput(text: string): Promise<void>`

Primary entry point. Submits user text, triggers triage, and runs the execution loop.

If called while already processing, automatically calls `interrupt()` with the new text.

```typescript
await agent.submitInput("Analyze the project structure");
```

#### `interrupt(newDirectives?: string): void`

Halts the current execution loop. Optionally injects new directives into the context.

```typescript
agent.interrupt("Stop, focus on the auth module instead");
```

#### `on(type: AgentEventType, handler: (event: AgentEvent) => void): () => void`

Subscribe to agent events. Returns an unsubscribe function.

```typescript
const unsub = agent.on("agent_thought", (event) => {
  console.log(event.payload.text);
});

// Later:
unsub();
```

#### `once(type: AgentEventType, handler: (event: AgentEvent) => void): () => void`

Subscribe for a single event emission.

```typescript
agent.once("turn_complete", (event) => {
  console.log("Turn done:", event.payload);
});
```

#### `getSession(): Session`

Returns the current as-agent Session object (append-only message history).

#### `processing: boolean` (getter)

Whether the agent is currently executing.

#### `removeAllListeners(): void`

Remove all event subscribers.

---

## AgentEventBus

Platform-agnostic typed event emitter. Used internally by `ObotoAgent` but also exported for custom use.

```typescript
import { AgentEventBus } from "@sschepis/oboto-agent";

const bus = new AgentEventBus();
bus.on("user_input", (event) => console.log(event));
bus.emit("user_input", { text: "hello" });
```

### Methods

- `on(type, handler)` — Subscribe, returns unsubscribe function
- `off(type, handler)` — Unsubscribe
- `once(type, handler)` — Subscribe for one emission
- `emit(type, payload)` — Emit event to all subscribers
- `removeAllListeners()` — Clear all subscriptions

---

## Adapter Functions

### `createRouterTool(router, root?)`

Bridges a swiss-army-tool `Router` into an lmscript `ToolDefinition`.

```typescript
import { createRouterTool } from "@sschepis/oboto-agent";
import { Router, TreeBuilder, SessionManager } from "@sschepis/swiss-army-tool";

const { root } = new TreeBuilder().branch("fs", "File system", (b) => {
  b.leaf("read", "Read a file", { path: "string" }, (args) => readFile(args.path));
}).build();

const router = new Router(root, new SessionManager("s1"));
const tool = createRouterTool(router, root);
// tool.name === "terminal_interface"
```

### `toChat(msg: ConversationMessage): ChatMessage`

Convert an as-agent message to lmscript format.

### `fromChat(msg: ChatMessage): ConversationMessage`

Convert an lmscript message to as-agent format.

### `sessionToHistory(session: Session): ChatMessage[]`

Convert an entire session to lmscript message array.

### `createEmptySession(): Session`

Create a fresh empty session (`{ version: 1, messages: [] }`).

---

## ContextManager

Manages the sliding context window with automatic summarization.

```typescript
import { ContextManager } from "@sschepis/oboto-agent";
import { LScriptRuntime, OllamaProvider } from "@sschepis/lmscript";

const runtime = new LScriptRuntime({
  provider: new OllamaProvider({ baseUrl: "http://localhost:11434" }),
});

const ctx = new ContextManager(runtime, "llama3:8b", 8192);
await ctx.push({ role: "user", content: "Hello" });
console.log(ctx.getMessages());
console.log(ctx.getTokenCount());
```

### Methods

- `push(message)` — Append a message, auto-prunes if over budget
- `pushAll(messages)` — Append multiple messages
- `getMessages()` — Get current context window
- `getTokenCount()` — Estimated token count
- `clear()` — Clear all context

---

## Triage

### `createTriageFunction(modelName: string)`

Creates an `LScriptFunction` for local-LLM triage classification.

Returns `{ escalate: boolean, reasoning: string, directResponse?: string }`.

### `TriageSchema`

The Zod schema used for triage validation. Can be used independently.

---

## Types

### AgentEventType

```typescript
type AgentEventType =
  | "user_input"
  | "agent_thought"
  | "triage_result"
  | "tool_execution_start"
  | "tool_execution_complete"
  | "state_updated"
  | "interruption"
  | "error"
  | "turn_complete";
```

### AgentEvent

```typescript
interface AgentEvent<T = unknown> {
  type: AgentEventType;
  payload: T;
  timestamp: number;
}
```

### TriageResult

```typescript
interface TriageResult {
  escalate: boolean;
  reasoning: string;
  directResponse?: string;
}
```

### ToolExecutionEvent

```typescript
interface ToolExecutionEvent {
  command: string;
  kwargs: Record<string, unknown>;
  result?: string;
  error?: string;
  durationMs?: number;
}
```
