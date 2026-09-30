/**
 * Alpha Step 4 scaling verification — `bun run alpha:verify-scaling`.
 *
 * A thin executable around `src/alpha/cli/verify-scaling.ts`, where the logic
 * lives. It runs a real training milestone on a model larger than Alpha's
 * 128,768-parameter starting point, over a corpus larger than the seed, and
 * verifies dataset versioning, data quality, tokenizer, parameter counting,
 * resource estimation, training, checkpointing, evaluation, the registry,
 * lifecycle promotion, export/import, inference and reproducibility.
 */

import { main } from "../src/alpha/cli/verify-scaling";

process.exit(await main(process.argv.slice(2)));
