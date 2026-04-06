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
});

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
  const schema = generateToolSchema({ root });

  return {
    name: schema.name,
    description: schema.description,
    parameters: RouterToolParams,
    execute: async (params) => {
      const cmd = typeof params.command === "string" ? params.command : "";
      return router.execute(cmd, params.kwargs ?? {});
    },
  };
}
