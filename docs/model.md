# Alpha Model

Alpha's language model is a **decoder-only transformer** implemented in
`src/alpha/model/transformer.ts` on top of Alpha's own autodiff engine. There
are no downloaded weights, no import from a model hub, and no conversion script
pointing at someone else's checkpoint. What exists is an architecture, a
configuration system, and a training engine that fills the parameters in.

---

## Architecture

```
token ids [B, T]
  → token embedding                [V, C] plus positional signal [T, C]
  → N × transformer block
        layer norm
        multi-head causal self-attention
          q = x·Wq + bq   k = x·Wk + bk   v = x·Wv + bv      (each [C, C])
          split into H heads of width C/H
          scores = (q · kᵀ) / √(C/H), causally masked, softmax
          context = attention · v, heads merged back to [B, T, C]
          output projection Wo
        residual add
        layer norm
        feed-forward: up-projection [C, dFF] → GELU → down-projection [dFF, C]
        residual add
  → final layer norm
  → output projection [C, V]        (tied to the token embedding by default)
  → logits [B·T, V]
```

Details that matter when reading the code:

- **Causal masking** is applied inside `causalSoftmax`: a position can attend to
  itself and everything before it, never after. A test verifies that changing
  the final token leaves earlier logits bit-for-bit identical.
- **Weight tying** shares the token embedding matrix with the output projection
  by default; `tieEmbeddings: false` adds an independent matrix.
- **Positional signal** is a learned embedding by default, with a fixed
  sinusoidal table available via `positionalEncoding: "sinusoidal"`.
- **Normalisation** is post-norm (layer norm before each sub-block) with
  `normEps` configurable per model.
- **Parameter names are stable strings** (`layer0.attn.wq`, `layer0.norm1.weight`,
  …). Checkpoints depend on them, and so does the workspace's architecture table.
- **Context overflow raises.** A sequence longer than `contextLength` throws
  `AlphaValidationError` rather than silently wrapping or padding.

---

## Configuration

```ts
import { createModelConfig, ALPHA_MODEL_PRESETS } from "@/alpha";

const config = createModelConfig({
  preset: "micro",
  contextLength: 96,
  dropout: 0.05,
});
```

| Field | Meaning |
| --- | --- |
| `name` / `version` | identity of the architecture record |
| `vocabSize` | must be ≥ the trained tokenizer's vocabulary |
| `contextLength` | maximum tokens the model will accept |
| `dModel` | residual stream width; must divide by `nHeads` |
| `nHeads` | attention heads; head width is `dModel / nHeads` |
| `nLayers` | transformer blocks |
| `dFeedForward` | up-projection width (typically 4× `dModel`) |
| `dropout` | applied inside blocks during training only |
| `normEps` | layer-norm epsilon |
| `positionalEncoding` | `learned` or `sinusoidal` |
| `tieEmbeddings` | share the embedding matrix with the output projection |
| `initStd` | standard deviation of the normal initialisation |

`createAlphaConfig({ preset, model: {…} })` in `src/alpha/configs` validates the
whole stack at once: tokenizer vocabulary ≤ model vocabulary, training sequence
length ≤ context length, retrieval chunk size ≤ context length.

### Presets

| Preset | Layers | Width | Heads | Feed-forward | Context | Parameters (before the vocabulary is fixed) |
| --- | --- | --- | --- | --- | --- | --- |
| `nano` | 2 | 64 | 4 | 256 | 64 | ~74k at vocab 384 |
| `micro` | 3 | 96 | 6 | 384 | 96 | ~190k at vocab 768 |
| `small` | 4 | 128 | 8 | 512 | 128 | ~700k at vocab 2048 |

`countParameters(config)` computes the number exactly, and
`describeArchitecture(config)` returns the same total as a table of tensor
shapes. Tests assert both equal the sum of the instantiated tensors, so the
number in the UI is never an estimate.

---

## Model stages

Alpha separates five things that are often conflated:

| Stage | What exists | How Alpha gets there |
| --- | --- | --- |
| `architecture` | config + code, no instantiated weights | configuration is defined |
| `untrained` | instantiated tensors, random values | `new AlphaTransformer(config)` |
| `trained` | a checkpoint from Alpha's own training run | `AlphaTrainer` completes a run |
| `fine-tuned` | a trained model continued on a narrower corpus | `isFineTune: true` on the run |
| `production` | a human promoted a trained model | explicit promotion |

`deriveStage()` implements this from artefacts (`hasWeights`, `hasCheckpoint`,
`trainedTokens`, `isFineTune`, `promoted`) — never from a flag someone sets by
hand. The workspace shows a stage badge, and the inference engine attaches the
stage and a warning to every generation, so untrained output is labelled at the
point of use as well.

---

## Parameter budget

For one block at width `C`, feed-forward `F`:

```
attention      4·C²        (Wq, Wk, Wv, Wo)
attention bias 4·C
feed-forward   2·C·F       (up and down projections)
ff biases      F + C
layer norms    4·C         (two norms, weight + bias)
```

Plus `V·C` for the token embedding, `T·C` for learned positions (when used),
`2·C` for the final norm, and `V·(C+1)` for an untied output projection.

Total for the `nano` preset at the seed corpus vocabulary (≈210 tokens) is
roughly 50k parameters — small enough to train a visible loss curve in a browser
tab in seconds.

---

## Versioning

Architecture version and weights are versioned separately:

- `config.version` — the architecture revision. Bump it when shapes or defaults
  change.
- `AlphaModelArtifact.version` — the model record's version, written to
  `alphaModels`.
- `AlphaCheckpoint.modelVersion` — which architecture a checkpoint's weights
  belong to. `AlphaWorkspace.resumeFrom()` refuses a checkpoint whose vocabulary
  does not match the current model, because silently loading mismatched weights
  would mean nothing at all.

---

## Embeddings from the model

`embedHidden(ids, batch, seq)` mean-pools the final layer-normed hidden states
into one vector per sequence. The embedder L2-normalises it, so similarity is a
dot product. This is deliberately the *same* encoder used for generation: Alpha
does not have a separate retrieval model, and pretending otherwise would be a
fiction. See [`docs/inference.md`](inference.md#embeddings) for the honest
statement about embedding quality.
