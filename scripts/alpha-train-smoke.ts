/**
 * Alpha training smoke run — `bun run alpha:train`.
 *
 * A thin executable around `src/alpha/cli/train-smoke.ts`, which is where the
 * logic (and the types) live. It runs Alpha's real training lifecycle on the
 * repository's own corpus and prints the measured result.
 */

import { main } from "../src/alpha/cli/train-smoke";

process.exit(main(process.argv.slice(2)));
