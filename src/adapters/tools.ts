import { z } from "zod";
import type { Router } from "@sschepis/swiss-army-tool";
import { generateToolSchema } from "@sschepis/swiss-army-tool";
import type { ToolDefinition } from "@sschepis/lmscript";
import type { BranchNode } from "@sschepis/swiss-army-tool";

/** Parameter schema for the omni-tool bridge. */
const RouterToolParams = z.object({
  command: z.string().describe(
    "The command or menu path (e.g., 'help', 'filesystem read', 'db query')"
  ),
  kwargs: z
    .record(z.unknown())
    .optional()
    .default({})
    .describe("Key-value arguments for the command"),
}).passthrough();

/**
 * Bridge a swiss-army-tool Router into an lmscript ToolDefinition.
 *
 * The LLM sees a single tool ("terminal_interface") with `command` and `kwargs`
 * parameters. When called, it routes through the swiss-army-tool command tree.
 */
export function createRouterTool(
  router: Router,
  root?: BranchNode
): ToolDefinition<typeof RouterToolParams, string> {
  const effectiveRoot = root ?? (router as any).root as BranchNode | undefined;
  const schema = generateToolSchema({ root: effectiveRoot });

  return {
    name: schema.name,
    description: schema.description,
    parameters: RouterToolParams,
    execute: async (params) => {
      const cmd = typeof params.command === "string" ? params.command : "";
      let kw = (params.kwargs != null && typeof params.kwargs === 'object' && !Array.isArray(params.kwargs))
        ? params.kwargs
        : {};
      // LLMs sometimes send tool args at the top level instead of nested in kwargs.
      // Merge any extra top-level keys into kwargs so they reach the handler.
      const { command: _c, kwargs: _k, ...rest } = params as Record<string, unknown>;
      if (Object.keys(rest).length > 0 && Object.keys(kw).length === 0) {
        kw = rest;
      }
      return router.execute(cmd, kw);
    },
  };
}
