# Architecture

## Overview

`oboto-agent` is an event-driven orchestration library that binds three specialized primitives into a coherent AI agent system:

```
┌─────────────────────────────────────────────────────┐
│                    ObotoAgent                        │
│                  (Orchestrator)                      │
│                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Event    │  │   Context    │  │    Triage     │  │
│  │   Bus     │  │   Manager    │  │   Function    │  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  Memory Adapter   │  │     Tools Adapter        │  │
│  │ (as-agent ↔       │  │ (swiss-army-tool →       │  │
│  │  lmscript)        │  │  lmscript ToolDef)       │  │
│  └──────────────────┘  └──────────────────────────┘  │
└──────┬──────────────────┬──────────────────┬─────────┘
       │                  │                  │
       ▼                  ▼                  ▼
  ┌──────────┐     ┌──────────────┐   ┌──────────────┐
  │ as-agent │     │   lmscript   │   │ swiss-army-  │
  │ (State)  │     │  (LLM I/O)  │   │ tool (Tools) │
  └──────────┘     └──────────────┘   └──────────────┘
```

## Dual-LLM Architecture

The agent uses two LLM providers:

1. **Local Model** (small, fast — e.g., Llama 3 8B via Ollama)
   - Triage: classifies user input as simple or complex
   - Summarization: compresses conversation history when context overflows
   - Direct responses: handles simple queries without network latency

2. **Remote Model** (powerful — e.g., Claude, GPT-4, Gemini)
   - Complex reasoning and multi-step tasks
   - Tool-calling loops with the swiss-army-tool Router
   - Code generation and analysis

## Execution Flow

```
User Input
    │
    ▼
┌─────────────┐
│ submitInput()│
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Record in session   │  ← as-agent Session (append-only)
│  + context manager   │  ← lmscript ContextStack
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Triage (local LLM) │  ← LScriptFunction via local runtime
└──────┬──────────────┘
       │
       ├── Simple ──→ Direct response ──→ turn_complete
       │
       └── Complex ──→ Escalate to remote model
                          │
                          ▼
                   ┌──────────────────┐
                   │  AgentLoop.run() │  ← lmscript AgentLoop
                   │  with tools      │  ← swiss-army-tool Router
                   └──────┬───────────┘
                          │
                          ├── Tool call ──→ Router.execute()
                          │                     │
                          │                     └── Result back to LLM
                          │
                          └── Final response ──→ turn_complete
```

## Event Bus

All state transitions emit typed events:

| Event | When | Payload |
|---|---|---|
| `user_input` | User submits text | `{ text }` |
| `triage_result` | Local LLM classifies input | `{ escalate, reasoning, directResponse? }` |
| `agent_thought` | LLM produces text output | `{ text, model, iteration? }` |
| `tool_execution_start` | Tool call begins | `{ command, kwargs }` |
| `tool_execution_complete` | Tool call finishes | `{ command, kwargs, result }` |
| `state_updated` | Session/context changes | `{ reason }` |
| `interruption` | User interrupts mid-loop | `{ newDirectives? }` |
| `error` | Something fails | `{ message, error }` |
| `turn_complete` | Full turn finishes | `{ model, escalated, iterations?, toolCalls? }` |

## Adapter Layer

### Memory Adapter (`adapters/memory.ts`)

Converts between as-agent's message format and lmscript's:

- **as-agent**: `ConversationMessage` with `MessageRole` enum and `ContentBlock[]` (kind: "text" / "tool_use" / "tool_result")
- **lmscript**: `ChatMessage` with `Role` string and `MessageContent` (string or content array)

### Tools Adapter (`adapters/tools.ts`)

Bridges swiss-army-tool's hierarchical command tree into a single lmscript `ToolDefinition`:

- The LLM sees one tool called `terminal_interface` with `{ command, kwargs }` parameters
- When called, `Router.execute(command, kwargs)` resolves through the command tree
- `generateToolSchema()` auto-generates the JSON schema including available modules

## Context Management

The `ContextManager` wraps lmscript's `ContextStack` with:

- Token budget enforcement (default 8192 tokens)
- Automatic pruning via the `summarize` strategy
- Summarization powered by the local LLM (no network calls needed)

When the context window fills up, the oldest non-system messages are summarized into a dense system message, preserving key context while freeing token budget.

## Interruption Handling

When the user submits new input while the agent is processing:

1. `submitInput()` detects `isProcessing === true` and calls `interrupt()`
2. The `interrupted` flag is set
3. `onIteration` and `onToolCall` callbacks in `AgentLoop` check the flag and return `false` to halt
4. The new directive is appended to the session and context
5. The user can then call `submitInput()` again with fresh context

Note: if an HTTP request to the LLM provider is in-flight, it will complete but its result is discarded.
