/**
 * Alpha runtime verification — `bun run alpha:verify-runtime`.
 *
 * A thin executable around `src/alpha/cli/verify-runtime.ts`, where the logic
 * lives. It trains a real model and then verifies every Step 3 subsystem with
 * real measurements: the KV cache, embeddings, vector ownership, memory,
 * context budgeting, retrieval, tools, agents and the orchestrator.
 */

import { main } from "../src/alpha/cli/verify-runtime";

process.exit(await main(process.argv.slice(2)));
