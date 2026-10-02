/**
 * Alpha Step 5 capability verification — `bun run alpha:verify-capability`.
 *
 * A thin executable around `src/alpha/cli/verify-capability.ts`, where the
 * logic lives. It walks the required pipeline end to end:
 *
 *   DATA -> TRAINING -> FROZEN EVALUATION -> MEASUREMENTS
 *         -> BASELINE COMPARISON -> CAPABILITY GATE -> REPORT
 *
 * Two copies of Alpha's own transformer are trained inside the run — one on the
 * Step 4 corpus, one on the Step 5 mixture — under one architecture, one
 * tokenizer and one training configuration, so the difference between them is
 * the corpus. Both are then measured against the same frozen evaluation suite
 * and compared measurement by measurement, with no composite score and no
 * declared winner. No external model is involved at any point.
 *
 * Exits 0 when every verification check passed, 1 otherwise.
 */

import { main } from "../src/alpha/cli/verify-capability";

process.exit(await main(process.argv.slice(2)));
