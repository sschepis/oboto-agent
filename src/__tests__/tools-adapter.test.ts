import { describe, it, expect, vi } from "vitest";
import {
  BranchNode,
  LeafNode,
  Router,
  SessionManager,
} from "@sschepis/swiss-army-tool";
import { createRouterTool } from "../adapters/tools.js";

function buildTestRouter() {
  const root = new BranchNode({ name: "root", description: "Root" });
  root.addChild(
    new LeafNode({
      name: "greet",
      description: "Say hello",
      handler: (kwargs) => `Hello, ${kwargs.name ?? "world"}!`,
      optionalArgs: { name: { type: "string", description: "Name to greet" } },
    })
  );
  root.addChild(
    new LeafNode({
      name: "add",
      description: "Add two numbers",
      requiredArgs: {
        a: { type: "number", description: "First number" },
        b: { type: "number", description: "Second number" },
      },
      handler: (kwargs) => String(Number(kwargs.a) + Number(kwargs.b)),
    })
  );

  const session = new SessionManager("test");
  const router = new Router(root, session);
  return { root, router };
}

describe("createRouterTool", () => {
  it("creates a tool with terminal_interface name", () => {
    const { router } = buildTestRouter();
    const tool = createRouterTool(router);

    expect(tool.name).toBe("terminal_interface");
    expect(tool.description).toBeTypeOf("string");
    expect(tool.description.length).toBeGreaterThan(0);
  });

  it("enriches description when root is provided", () => {
    const { router, root } = buildTestRouter();
    const tool = createRouterTool(router, root);

    expect(tool.description).toContain("greet");
    expect(tool.description).toContain("add");
  });

  it("executes commands through the router", async () => {
    const { router } = buildTestRouter();
    const tool = createRouterTool(router);

    const result = await tool.execute({ command: "greet", kwargs: { name: "Alice" } });
    expect(result).toContain("Hello, Alice!");
  });

  it("executes commands with required args", async () => {
    const { router } = buildTestRouter();
    const tool = createRouterTool(router);

    const result = await tool.execute({ command: "add", kwargs: { a: 3, b: 7 } });
    expect(result).toContain("10");
  });

  it("executes with empty kwargs by default", async () => {
    const { router } = buildTestRouter();
    const tool = createRouterTool(router);

    const result = await tool.execute({ command: "greet", kwargs: {} });
    expect(result).toContain("Hello, world!");
  });
});
