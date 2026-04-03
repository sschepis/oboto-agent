# oboto-agent

Event-driven dual-LLM orchestration library for autonomous AI agents.

`oboto-agent` is a lightweight TypeScript library that acts as the central nervous system for AI agents. It binds three specialized primitives together through a typed event bus:

- **[lmscript](https://github.com/sschepis/lmscript)** — LLM I/O, structured output, provider abstraction
- **[swiss-army-tool](https://github.com/sschepis/swiss-army-tool)** — Hierarchical tool execution
- **[as-agent](https://github.com/sschepis/as-agent)** — Session state and conversation history

## Key Features

- **Dual-LLM architecture** — Fast local model (Ollama/LMStudio) for triage, powerful cloud model (Anthropic/OpenAI/Gemini) for complex tasks
- **Automatic triage** — Local model classifies each input and only escalates when needed
- **Event-driven** — All state transitions emit typed events for CLI, web, or daemon integration
- **Context management** — Automatic summarization when context window fills up
- **Interruption handling** — Users can redirect the agent mid-execution
- **Platform-agnostic** — No Node.js-specific APIs; works in browser, Deno, and Bun
- **Headless** — No UI framework dependency; bring your own interface

## Installation

```bash
npm install @sschepis/oboto-agent

# Peer dependencies
npm install @sschepis/lmscript @sschepis/swiss-army-tool @sschepis/as-agent zod
```

## Quick Start

```typescript
import { ObotoAgent } from "@sschepis/oboto-agent";
import { OllamaProvider, AnthropicProvider } from "@sschepis/lmscript";
import { TreeBuilder, Router, SessionManager } from "@sschepis/swiss-army-tool";

// Build tools
const builder = new TreeBuilder();
builder.leaf("time", "Get current time", {}, () => new Date().toISOString());
builder.leaf("greet", "Say hello", { name: "string" }, (args) => `Hello, ${args.name}!`);
const { root } = builder.build();
const router = new Router(root, new SessionManager("s1"));

// Create agent
const agent = new ObotoAgent({
  localModel: new OllamaProvider({ baseUrl: "http://localhost:11434" }),
  remoteModel: new AnthropicProvider({ apiKey: process.env.ANTHROPIC_API_KEY! }),
  localModelName: "llama3:8b",
  remoteModelName: "claude-sonnet-4-20250514",
  router,
});

// Listen to events
agent.on("agent_thought", (e) => console.log(e.payload.text));
agent.on("tool_execution_complete", (e) => console.log("Tool:", e.payload.result));
agent.on("error", (e) => console.error(e.payload.message));

// Run
await agent.submitInput("What time is it?");
```

## Architecture

```
User Input
    │
    ▼
┌──────────────────────────┐
│       ObotoAgent          │
│                           │
│  ┌─────────┐ ┌────────┐  │
│  │ Event   │ │Context │  │
│  │  Bus    │ │Manager │  │
│  └─────────┘ └────────┘  │
│                           │
│  Triage (local LLM)      │
│    ├── Simple → respond   │
│    └── Complex → escalate │
│                           │
│  AgentLoop (remote LLM)  │
│    └── Tool calls → Router│
└──────────────────────────┘
    │         │         │
    ▼         ▼         ▼
 as-agent  lmscript  swiss-army-tool
 (state)   (LLM I/O)  (tools)
```

### Execution Flow

1. **Input** — User submits text via `submitInput()`
2. **Record** — Message appended to session (as-agent) and context window (lmscript ContextStack)
3. **Triage** — Local LLM classifies: simple or complex?
4. **Direct response** — If simple, local model responds immediately
5. **Escalate** — If complex, remote model runs with tool access via AgentLoop
6. **Tool execution** — LLM calls tools through the swiss-army-tool Router
7. **Turn complete** — Response recorded, events emitted

### Events

| Event | Description |
|---|---|
| `user_input` | User submitted text |
| `triage_result` | Local LLM classified the input |
| `agent_thought` | LLM produced text output |
| `tool_execution_start` | Tool call began |
| `tool_execution_complete` | Tool call finished |
| `state_updated` | Session or context changed |
| `interruption` | User interrupted mid-execution |
| `error` | Something failed |
| `turn_complete` | Full turn finished |

## API

### `ObotoAgent`

```typescript
const agent = new ObotoAgent(config: ObotoAgentConfig);

await agent.submitInput(text);        // Submit user input
agent.interrupt(newDirectives?);      // Halt and redirect
agent.on(event, handler);             // Subscribe (returns unsub fn)
agent.once(event, handler);           // One-time subscribe
agent.getSession();                   // Get session state
agent.processing;                     // Is currently executing?
agent.removeAllListeners();           // Clear all subscriptions
```

### Configuration

```typescript
interface ObotoAgentConfig {
  localModel: LLMProvider;       // Fast local model
  remoteModel: LLMProvider;      // Powerful cloud model
  localModelName: string;        // e.g. "llama3:8b"
  remoteModelName: string;       // e.g. "claude-sonnet-4-20250514"
  router: Router;                // swiss-army-tool Router
  session?: Session;             // Resume existing session
  maxContextTokens?: number;     // Default: 8192
  maxIterations?: number;        // Default: 10
  systemPrompt?: string;         // Custom system prompt
}
```

### Utility Exports

```typescript
// Adapters
createRouterTool(router, root?)    // Router → lmscript ToolDefinition
toChat(msg)                         // as-agent → lmscript message
fromChat(msg)                       // lmscript → as-agent message
sessionToHistory(session)           // Session → ChatMessage[]
createEmptySession()                // Fresh empty session

// Components
AgentEventBus                       // Standalone event emitter
ContextManager                      // Context window manager
createTriageFunction(modelName)     // Triage LScriptFunction
TriageSchema                        // Zod schema for triage output
```

## Examples

See the [examples/](examples/) directory:

- **[basic-cli.ts](examples/basic-cli.ts)** — Interactive CLI agent with triage
- **[custom-tools.ts](examples/custom-tools.ts)** — Rich file system tool tree
- **[event-monitoring.ts](examples/event-monitoring.ts)** — Log all agent events with colors
- **[session-persistence.ts](examples/session-persistence.ts)** — Save/restore sessions across restarts

## Documentation

- **[Architecture](docs/architecture.md)** — System design, data flow, adapter layer
- **[API Reference](docs/api.md)** — Complete API documentation
- **[Guides](docs/guides.md)** — How-to guides for common tasks

## Development

```bash
npm install         # Install dependencies
npm run build       # Build with tsup
npm test            # Run tests with vitest
npm run typecheck   # Type-check without emitting
npm run dev         # Watch mode build
```

## Supported Providers

Any lmscript provider works as either the local or remote model:

| Provider | Package | Typical Role |
|---|---|---|
| Ollama | `OllamaProvider` | Local |
| LM Studio | `LMStudioProvider` | Local |
| Anthropic | `AnthropicProvider` | Remote |
| OpenAI | `OpenAIProvider` | Remote |
| Google Gemini | `GeminiProvider` | Remote |
| OpenRouter | `OpenRouterProvider` | Remote |
| DeepSeek | `DeepSeekProvider` | Remote |
| AWS Bedrock | `VertexAnthropicProvider` | Remote |

## License

MIT
