# Alpha Training

Alpha trains by backpropagation through its own autodiff engine. There is no
distillation step, no weight download and no external trainer: the corpus is
tokenised, batched, passed forward, scored with cross-entropy, scored backward,
and applied to the parameters with AdamW.

The pipeline is not described here as if it were intended — it has been run in
this repository, and the numbers further down are what that run printed. See
[Measured run](#measured-run).

```
src/alpha/datasets   corpus, splitting, batching
src/alpha/tokenizer  vocabulary training
src/alpha/training   optimiser, schedule, checkpoints, trainer, job record, verification
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
  id; label; formatVersion;
  runId;                 // the training run that wrote it
  seed;                  // the run's sampling seed
  trainingConfig;        // the config the run started with
  modelName; modelVersion; config; configFingerprint;
  tokenizer: {           // enough to rebuild the vocabulary, not just name it
    version; vocabSize; fingerprint; trainedOn; specialTokenIds;
    snapshot;            // full merge table + alphabet
  };
  tokenizerVersion;      // convenience mirror
  datasetName; datasetVersion; datasetFingerprint; datasetLicense;
  step; tokensSeen; learningRate;
  metrics: { trainLoss, validationLoss, validationPerplexity, uniformLoss };
  weights: SerializedWeights;          // base64 float32 per parameter
  optimizer: OptimizerStateSnapshot;   // Adam first/second moments
  rng: RngState;                       // exact sampler position
  createdAt; sizeBytes; stage; isFineTune; notes;
};
```

The checkpoint carries the **tokenizer snapshot**, not just its version. That is
deliberate: a checkpoint plus its own tokenizer is enough to rebuild the exact
model that produced it, with no dependency on whatever vocabulary happens to be
loaded later.

Weights are stored as base64 float32 rather than JSON numbers: a checkpoint is
compact enough to persist in Alpha's own tables and to check into a fixture. The
`nano` run's checkpoint is 515,072 bytes of weights — 2,072,746 bytes of JSON
once base64 is expanded and every field is spelled out.

`validateCheckpoint` / `assertValidCheckpoint` check the payload's structure,
`parseCheckpoint` re-validates on the way back out of storage and throws
`AlphaCheckpointError` on anything malformed, and `assertCheckpointCompatible`
compares the architecture fingerprint, the tokenizer fingerprint and the
vocabulary size against the live model before a single weight is loaded.

**Resuming is real.** `trainer.resumeFrom(checkpoint)` restores the weights, the
optimiser moments (so Adam's running estimates are not reset) and the RNG
position, then continues from the recorded step:

```ts
trainer.resumeFrom(stored);        // step is now stored.step
trainer.trainToCompletion();       // continues instead of restarting
```

Because the checkpoint stores the run's **training config**, a resumed run
continues toward the same step budget on the same schedule rather than picking up
whatever the workspace currently defaults to. `resumes` is counted on the job
record, so "resumed once" and "trained from scratch" stay distinguishable.

The workspace exposes the same operation against a checkpoint loaded from
Convex, and rejects a checkpoint whose vocabulary, architecture fingerprint or
tokenizer fingerprint does not match the current model — loading mismatched
weights would silently mean nothing.

---

## 6. The run record

Every run has an `AlphaTrainingJob` beside the metrics: state, seed, references,
progress, and what happened to it.

```ts
type AlphaTrainingJob = {
  id; state;                 // created | running | paused | completed | failed | stopped
  model: { name, version, configFingerprint };
  tokenizer: { version, vocabSize, fingerprint, license };
  dataset: { name, version, fingerprint, license, tokens, documents };
  config; seed; step; totalSteps; epochs; tokensSeen;
  trainLoss; bestLoss; validationLoss; learningRate;
  checkpointIds; lastCheckpointId;
  resumedFromCheckpointId; resumes;
  createdAt; startedAt; updatedAt; finishedAt; error; notes;
};
```

State changes go through `transitionJob`, which enforces the legal graph
(`canTransitionJob`) — a paused job cannot jump straight to completed, and a
failed job does not silently restart. `createTrainingJob`, `recordJobStep`,
`recordJobEvaluation`, `recordJobCheckpoint` and `failJob` are the only ways the
record changes, so it cannot drift away from what actually happened.
`summariseJob(job)` renders the one-line form the UI shows:

```
run_muo21cftgqm · RUNNING · step 34/60 · train 4.1021 · val 4.3187 · 1 checkpoint(s)
```

The workspace exposes pause, resume-from-store and stop, and each writes the
transition it performed. Verification trains a **fresh** model instance, so
pressing *Verify* never disturbs the weights of a run in progress.

---

## 7. Verification

`verifyAlphaModel()` is the answer to "does this thing actually work?" — and it
answers with measurements rather than a claim.

```ts
import { verifyAlphaModel } from "@/alpha";

const report = verifyAlphaModel({ model, tokenizer, dataset, training: { totalSteps: 40 } });
report.passed;                 // true only if all nine checks pass
report.checks;                 // [{ id, label, passed, detail, data }]
report.training;               // firstLoss, lastLoss, uniformLoss, validationLoss, …
```

| Check | What it proves |
| --- | --- |
| A | initial parameters are not all identical (a real distribution, not a constant) |
| B | one real optimiser step moves the weights — the weight hash changes |
| C | gradients are non-zero **and** match central differences of the loss |
| D | the reported loss equals an independent softmax/NLL recomputation, and differs per batch |
| E | a checkpoint can be written and passes validation |
| F | that checkpoint can be parsed back and re-validated |
| G | every reloaded tensor matches the saved one — maximum absolute difference 0 |
| H | training resumes from the checkpoint and continues the same run |
| I | inference runs on the reloaded weights; repeat decode is identical and the first token is the model's own argmax |

Run it from the terminal:

```bash
bun run alpha:verify --steps 40
```

The measured output of that command in this repository:

```
Verification PASSED
  A. ok  Initial parameters are not all identical — Sampled 512 distinct values from the token embedding; range [-5.662e-2, 4.674e-2].
  B. ok  A training step changes parameters — Weights hash moved from w_0895ff11 to w_68f44bd7 after one real optimiser step (loss 5.9107).
  C. ok  Gradients are non-zero and match numerical differences — 2120/2304 sampled gradient entries are non-zero. Autodiff matches central differences on 72 sampled value(s) across 36 tensor(s); largest relative error above the 1e-4 absolute floor is 0.000e+0 (tolerance 0.05), largest absolute error 2.465e-5.
  D. ok  Loss is calculated from the model's actual predictions — Model loss 5.963971 equals an independent softmax/NLL recomputation (5.963971) over 16 tokens; a different batch gives 5.975486 and the uniform baseline is 5.950643.
  E. ok  A checkpoint can be saved — Wrote ckpt_muo21qdw1yzrd at step 8 (2,072,747 bytes of JSON, 515,072 bytes of weights).
  F. ok  The checkpoint can be reloaded — Parsed ckpt_muo21qdw1yzrd back from JSON and re-validated it (format 1.0.0, tokenizer tok_02690d5b).
  G. ok  Reloaded parameters match the saved parameters — Every tensor in the payload reloaded with a maximum absolute difference of exactly 0 across the whole parameter set.
  H. ok  Training can resume from the checkpoint — Restored step 8 and AdamW at step 8, then trained 2 more step(s) to 10 (loss 4.7614, validation 5.0102).
  I. ok  Inference runs on the trained/reloaded model and emits tokens — Greedy decode produced 12 token(s) from alpha-nano (max-tokens); repeating it gives identical ids: true. The first generated token 4 is the argmax 4 of the reloaded model's own logits (true).
```

---

## 8. Resource envelope

Training a model in a browser tab needs a bound, and Alpha states it before the
run rather than discovering it as a crash.

```ts
import { ALPHA_RESOURCE_LIMITS, estimateTrainingMemory, countParametersFromConfig } from "@/alpha";

const estimate = estimateTrainingMemory(config, { batchSize: 8, seqLen: 32 });
// { parameterCount, weightsBytes, gradientBytes, optimizerBytes,
//   activationBytes, totalBytes, note }
```

`assertResourceLimit(kind, value)` raises rather than letting a run die
mid-flight. The ceilings — `maxContextLength`, `maxVocabSize`, `maxLayers`,
`maxDModel`, `maxParameterCount`, `maxBatchSize`, `maxSeqLen`, `maxTotalSteps`,
`maxNewTokens`, `maxPromptTokens`, `maxDocuments`, `maxDocumentCharacters`,
`maxDropout` — are deliberately small: Alpha's stack is real but CPU-only. The
Training panel shows the estimate and the limits side by side, so a
configuration that would not fit is visible before you press *Train Alpha*.

---

## 9. Measured run

This is the actual output of `bun run alpha:train` (60 steps, `nano`, seed 1337)
in this repository, reproduced rather than approximated:

```
ALPHA LIFECYCLE REPORT — executed 2026-09-30T12:01:30.236Z in 8.88s
Overall: OK — every stage completed

Model        alpha-nano v0.1.0 (nano) · 128,768 params · 2L/64d/4h · context 64 · vocab 384
Tokenizer    v0.1.0 · 384 tokens · 333 merges · tok_02690d5b
Corpus       alpha-seed@1.0.0 (CC0-1.0 (authored for this repository)) · 20 docs · 5,358 chars · 2,634 tokens
Split        train 2,214 tokens / 69 examples · validation 420 tokens / 13 examples
Training     60 steps · batch 8 × seq 32 · seed 1337 · 15,360 tokens · 1905 tokens/s
Loss         first 5.9428 → last 3.9579 · best 3.8190 · uniform baseline 5.9506 · validation 4.2403
Checkpoint   ckpt_muo21inu01bmj (run run_muo21cftgqm) · step 60 · stage trained · 515,072 bytes · valid true
Reload       compatible true · max absolute weight difference 0
Resume       step 60 → 62 (+2) · loss after resume 3.8460
Inference    24 token(s) · stop max-tokens · stage trained · deterministic true
Token ids    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
```

What that establishes, in order:

1. **The tokenizer trained itself** on the shipped corpus — 333 merges, 384
   tokens, fingerprint `tok_02690d5b`.
2. **The corpus was encoded and split** deterministically: 2,214 training and 420
   validation tokens, zero unknown characters.
3. **Loss fell below the uniform baseline** — 5.9428 → 3.9579, best 3.8190,
   against `ln(384) = 5.9506`. Mean gradient norm 1.4702. Weights changed.
4. **A checkpoint was written and validated** at step 60: 515,072 bytes of
   weights, stage `trained`.
5. **The checkpoint reloaded exactly** — maximum absolute weight difference 0
   across the whole parameter set, with the architecture and tokenizer
   fingerprints checked first.
6. **Training resumed from it** — step 60 → 62, continuing the same run rather
   than restarting, loss 3.8460.
7. **Inference ran on the trained weights** and produced tokens, deterministically
   under greedy decoding, and the decoded text round-trips from the token ids.

**Read the last line honestly.** Twenty-four copies of token 4 (`"\n"`) is what
a 128,768-parameter model trained on twenty paragraphs can do. The lifecycle is
verified; the *language quality* is not, and it is not claimed. Alpha labels the
stage on every result so the two are never confused.

---

## 10. Running training

### From the terminal

```bash
bun run alpha:train                              # 60 steps, nano, seed 1337
bun run alpha:train --steps 200 --preset micro
bun run alpha:verify --steps 40                  # the nine checks
bun run alpha:train --json > run.json            # machine-readable
```

The CLI executes the real pipeline — corpus, tokenizer, transformer, loss,
backpropagation, AdamW, checkpoint, reload, resume, inference — and writes
nothing to disk. Confidence in Alpha comes from running this, not from trusting
a number in a document.

### In the browser workspace

`/dashboard` → **Training**. Set steps, batch, sequence length, learning rate
and validation interval, press *Train Alpha*. The loss curve updates while it
runs; a checkpoint is written to Convex at the interval you set, and the model
badge changes from `UNTRAINED` to `TRAINED (FROM SCRATCH)` when one exists.

The panel also shows the **run record** (run id, tokenizer and dataset
fingerprints, seed, step, tokens, epochs, losses, checkpoint ids, resume count),
the **verification report** (checks A–I with their measured details), the
**corpus report** (fingerprint, train/validation examples, unknown characters,
sequence length, batch size, padding) and the **resource envelope**. While a run
is in progress the action bar offers *Pause*, *Continue run* and *Stop* rather
than only *Train*.

### In a script

```ts
const trainer = new AlphaTrainer({ model, tokenizer, dataset });
const summary = trainer.trainToCompletion();
console.log(summary.lastLoss, summary.validationLoss, summary.uniformLossBaseline);
```

### What to expect

The seed corpus is tiny. A few dozen steps will move the loss down and beat the
uniform baseline; more steps overfit it. That is the correct result for
128,768 parameters on ~20 paragraphs, and it is reported rather than papered
over. Point Alpha at a real corpus and raise the preset to get a real model.

---

## 11. Configuration reference

| Field | Default | Notes |
| --- | --- | --- |
| `batchSize` | 8 | windows per step |
| `seqLen` | 32 | must be ≤ `contextLength` |
| `totalSteps` | 120 | optimiser steps in the run; a resumed run continues toward this number |
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
