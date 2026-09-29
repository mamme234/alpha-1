# Alpha Training

Alpha trains by backpropagation through its own autodiff engine. There is no
distillation step, no weight download and no external trainer: the corpus is
tokenised, batched, passed forward, scored with cross-entropy, scored backward,
and applied to the parameters with AdamW.

```
src/alpha/datasets   corpus, splitting, batching
src/alpha/tokenizer  vocabulary training
src/alpha/training   optimiser, schedule, checkpoints, trainer
src/alpha/core       tensor and autodiff used by all of the above
```

---

## 1. Corpus

A dataset is documents plus provenance:

```ts
type AlphaDataset = {
  id: string; name: string; version: string; description: string;
  license: string;        // SPDX-ish; travels into every checkpoint
  source: string;
  documents: string[];
};
```

The repository ships one: `ALPHA_SEED_CORPUS` in
`src/alpha/datasets/seed-corpus.ts`, twenty original paragraphs about Alpha's
own architecture, released CC0-1.0. It is deliberately repetitive in places —
a small model cannot learn much, but it can learn the shape of those sentences,
which makes the loss curve honest and readable.

`encodeCorpus(dataset, tokenizer, { validationFraction })` produces a train
stream and a validation stream. Documents are interleaved rather than sliced so
both halves see every style in the corpus, and each document is wrapped in
`<bos> … <eos>` so boundaries are learnable.

**Bring your own data.** Drop documents into a new `AlphaDataset`, point
`AlphaWorkspace({ dataset })` at it, and retrain. Keep the licence field
accurate: it is copied into every checkpoint your run produces.

---

## 2. Tokenizer training

The vocabulary must be trained before the model can be built, because the
model's `vocabSize` is set from it.

```ts
const tokenizer = AlphaTokenizer.train(dataset.documents, {
  vocabSize: 384,
  version: "0.1.0",
  trainedOn: `${dataset.name}@${dataset.version}`,
  minPairFrequency: 2,
});
```

Character-level BPE: start from the characters observed in the corpus, then
repeatedly merge the most frequent adjacent symbol pair until the target
vocabulary size or `minPairFrequency` stops the process. What you get back is a
portable artifact — the merge table, the alphabet, special tokens, stats and
version all serialise to JSON and reload exactly:

```ts
const snapshot = tokenizer.toJSON();
const reloaded = AlphaTokenizer.fromJSON(snapshot);   // identical token ids
```

Encoding applies merges by rank; decoding concatenates tokens. Characters
outside the trained alphabet become `<unk>` and are **counted**, so the trainer
can tell you how much of your corpus the vocabulary could not represent.

Special tokens are `<pad> <unk> <bos> <eos>`, allocated at fixed ids so ids stay
stable across vocabulary versions.

---

## 3. The training loop

```ts
const trainer = new AlphaTrainer({
  model, tokenizer, dataset,
  config: {
    batchSize: 8, seqLen: 32, totalSteps: 60,
    learningRate: 3e-3, schedule: "cosine", warmupSteps: 6, minFactor: 0.1,
    weightDecay: 0.01, gradClipNorm: 1,
    evalInterval: 15, evalBatches: 4,
    validationFraction: 0.12, seed: 1337,
    checkpointInterval: 30,
  },
});

for (const event of trainer.run()) {
  if (event.type === "step") console.log(event.point.step, event.point.loss);
  if (event.type === "checkpoint") save(event.checkpoint);
}
```

Each step is:

1. `BatchSampler.next()` draws random windows from the training stream; targets
   are the inputs shifted by one position.
2. `model.forward(..., { training: true })` runs the graph.
3. `crossEntropy(logits, targets, padId)` computes mean NLL, ignoring padding.
4. `backward(loss)` performs a topological reverse pass, accumulating grads.
5. `AdamW.stepWithSchedule(lr)` clips the global gradient norm, applies
   decoupled weight decay and updates every parameter.
6. The optimiser's gradients are zeroed and the metrics are recorded.

`run()` is a **generator**: each `next()` performs exactly one optimiser step.
That is what lets the browser train in slices without freezing, and it is why
the workspace can plot the curve while the run is still going.
`trainToCompletion()` is the synchronous helper for scripts and tests.

### Why the loop is trustworthy

- Gradients come from the same ops that were used in the forward pass; there is
  no separate hand-derived path that could drift.
- The test suite checks each op's analytic gradient against a central difference
  of the loss, then checks that a real run reduces validation loss below the
  uniform baseline `ln(vocabSize)`.
- Reported metrics are the trainer's own: `loss` is the cross-entropy just
  computed, `gradNorm` is the L2 norm before clipping, `updateNorm` is the size
  of the step actually applied.

### Learning-rate schedule

`learningRateAt(step, schedule)` with `constant`, `linear-decay`, `cosine`
(default) or `inverse-sqrt`. Warmup prevents divergence in the first dozen
steps; `minFactor` sets the floor. The schedule is a pure function of the step
number, so a resumed run reproduces the rate it would have used.

---

## 4. Evaluation

```ts
const evaluation = trainer.evaluate({ maxBatches: 4 });
// { loss, perplexity, batches, tokens, uniformLoss }
```

Validation uses `sequentialBatches()` for determinism and runs with gradient
recording disabled (`setGradEnabled(false)`), so evaluation is pure forward work
and leaves no graph behind. `uniformLoss` is `ln(vocabSize)` — the loss of a
predictor that guesses uniformly. Reporting it next to the real loss is what
makes a curve interpretable: until the model beats that line, it has learned
nothing useful.

---

## 5. Checkpoints and resuming

A checkpoint is the unit of *"Alpha has actually trained"*:

```ts
type AlphaCheckpoint = {
  id; label; modelName; modelVersion; config;
  tokenizerVersion; datasetName; datasetLicense;
  step; tokensSeen; learningRate;
  metrics: { trainLoss, validationLoss, validationPerplexity | null };
  weights: SerializedWeights;          // base64 float32 per parameter
  optimizer: OptimizerStateSnapshot;   // Adam first/second moments
  rng: RngState;                       // exact sampler position
  createdAt; sizeBytes; stage; notes;
};
```

Weights are stored as base64 float32 rather than JSON numbers: a checkpoint is
compact enough to persist in Alpha's own tables and to check into a fixture.

**Resuming is real.** `trainer.resumeFrom(checkpoint)` restores the weights, the
optimiser moments (so Adam's running estimates are not reset) and the RNG
position, then continues from the recorded step:

```ts
trainer.resumeFrom(stored);        // step is now stored.step
trainer.trainToCompletion();       // continues instead of restarting
```

The workspace exposes the same operation against a checkpoint loaded from
Convex, and rejects a checkpoint whose vocabulary does not match the current
model — loading mismatched weights would silently mean nothing.

---

## 6. Running training

### In the browser workspace

`/dashboard` → **Training**. Set steps, batch, sequence length, learning rate
and validation interval, press *Train Alpha*. The loss curve updates while it
runs; a checkpoint is written to Convex at the interval you set, and the model
badge changes from `UNTRAINED` to `TRAINED (FROM SCRATCH)` when one exists.

### In a script

```ts
const trainer = new AlphaTrainer({ model, tokenizer, dataset });
const summary = trainer.trainToCompletion();
console.log(summary.lastLoss, summary.validationLoss, summary.uniformLossBaseline);
```

### What to expect

The seed corpus is tiny. A few dozen steps will move the loss down and beat the
uniform baseline; more steps overfit it. That is the correct result for
~50k parameters on ~20 paragraphs, and it is reported rather than papered over.
Point Alpha at a real corpus and raise the preset to get a real model.

---

## 7. Configuration reference

| Field | Default | Notes |
| --- | --- | --- |
| `batchSize` | 8 | windows per step |
| `seqLen` | 32 | must be ≤ `contextLength` |
| `totalSteps` | 120 | optimiser steps in the run |
| `learningRate` | 3e-3 | peak rate before the schedule |
| `schedule` | `cosine` | `constant`, `linear-decay`, `cosine`, `inverse-sqrt` |
| `warmupSteps` | 12 | linear warmup |
| `minFactor` | 0.1 | floor as a fraction of peak |
| `weightDecay` | 0.01 | decoupled (AdamW) |
| `gradClipNorm` | 1 | global-norm clip; 0 disables |
| `evalInterval` | 20 | 0 disables periodic evaluation |
| `evalBatches` | 4 | validation batches per evaluation |
| `validationFraction` | 0.12 | share of documents held out |
| `seed` | 1337 | batch sampling and dropout |
| `checkpointInterval` | 60 | 0 disables periodic checkpoints |

See `configs/alpha.nano.json`, `configs/alpha.micro.json` and
`configs/alpha.small.json` for complete example configurations.
