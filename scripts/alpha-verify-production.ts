/**
 * Alpha production verification — `bun run alpha:verify-production`.
 *
 * A thin executable around `src/alpha/cli/verify-production.ts`, where the logic
 * lives. It verifies the exact modules the Convex chat API serves with, then
 * (unless `--offline` is passed) drives the live deployment through the Convex
 * CLI: sign-up, a streamed turn, persistence, stop, isolation between accounts,
 * an oversized request and a bad token.
 */

import { main } from "../src/alpha/cli/verify-production";

process.exit(await main(process.argv.slice(2)));
