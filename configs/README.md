# Configuration examples

These files are complete, validated Alpha configurations. They are the same
shape the code uses — `parseAlphaConfig(json)` merges them over the defaults and
throws if a combination is inconsistent (for example a tokenizer vocabulary
larger than the model vocabulary, or a training sequence longer than the model
context).

| File | Preset | Layers | Width | Context | Target vocabulary |
| --- | --- | --- | --- | --- | --- |
| `alpha.nano.json` | `nano` | 2 | 64 | 64 | 384 |
| `alpha.micro.json` | `micro` | 3 | 96 | 96 | 512 |
| `alpha.small.json` | `small` | 4 | 128 | 128 | 1024 |

## Using one

```ts
import { readFileSync } from "node:fs";
import { AlphaWorkspace, parseAlphaConfig } from "@/alpha";

const config = parseAlphaConfig(readFileSync("configs/alpha.nano.json", "utf8"));
const workspace = new AlphaWorkspace({ config }).constructor === AlphaWorkspace
  ? new AlphaWorkspace({
      config: {
        preset: config.preset,
        model: config.model,
        tokenizer: config.tokenizer,
        training: config.training,
        rag: config.rag,
        memory: config.memory,
        security: config.security,
        automation: config.automation,
        observability: config.observability,
      },
    })
  : null;
await workspace?.initialise();
```

Shorter form — pass the overrides object directly:

```ts
import raw from "../configs/alpha.micro.json";
import { createAlphaConfig, AlphaWorkspace } from "@/alpha";

const config = createAlphaConfig(raw);
const workspace = await new AlphaWorkspace({ config: raw }).initialise();
```

## Field notes

- **`model.vocabSize`** must be at least the trained vocabulary size. The
  workspace sets it from the tokenizer at initialisation, so treat the value in
  these files as a ceiling.
- **`tokenizer.targetVocabSize`** is a ceiling too: BPE stops early when no pair
  occurs at least `minPairFrequency` times.
- **`training.seqLen`** must be ≤ `model.contextLength`; the sampler draws random
  windows of that length.
- **`training.totalSteps`** is the whole run. Resuming from a checkpoint
  continues toward this number rather than restarting.
- **`rag.chunkTokens`** must be ≤ `model.contextLength`, and
  `overlapTokens` must be smaller than `chunkTokens`.
- **`security.sandbox`** defaults deny network and filesystem. Enabling either is
  a deliberate act; the agent scope and tool allow-list still apply on top.
- **`observability.logLevel`** is `debug` | `info` | `warn` | `error`. `debug`
  records per-step training metrics.

## Guarantees these files do not contain

No API keys, no provider endpoints, no model identifiers pointing at a hosted
service, and no secrets of any kind. Alpha has no external model dependency to
configure, so a configuration file is limited to the model, the corpus settings
and the rules Alpha runs under.
