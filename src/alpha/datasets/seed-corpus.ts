/**
 * Alpha Datasets — seed corpus.
 *
 * This text was written for this repository so that Alpha always has a corpus
 * it is legally allowed to train on. It is deliberately repetitive in places:
 * a 300k-parameter transformer cannot learn much, but it can learn the shape of
 * these sentences, which makes the training loss curve honest and readable.
 *
 * Replace or extend it with your own documents — the pipeline does not care
 * where the strings come from, only that the licence permits training.
 */

import { createDataset, type AlphaDataset } from "./types";

const documents: string[] = [
  `Alpha is a self owned artificial intelligence stack. The language model at the centre of Alpha is written in this repository and trained by this repository. Alpha does not call an outside provider for text generation, embeddings, or retrieval. When Alpha does not know something, it says so.`,

  `The Alpha language model is a decoder only transformer. Each block contains a layer norm, a multi head causal attention layer, a residual connection, a second layer norm, a feed forward network, and a second residual connection. The final layer norm feeds an output projection that shares weights with the token embedding.`,

  `Attention in Alpha is causal. A token may attend to itself and to every token before it, and never to a token after it. The attention scores are scaled by one over the square root of the head width, then normalised with a softmax, then multiplied by the value vectors.`,

  `Alpha trains with backpropagation. Gradients flow backwards through every operation: matrix multiplication, layer normalisation, softmax, the gelu activation, dropout, and the embedding lookup. The optimiser is AdamW with decoupled weight decay and gradient clipping.`,

  `The Alpha tokenizer is trained on Alpha's own corpus. It begins with the characters it observes, then merges the most frequent adjacent pair again and again until the vocabulary is full. The tokenizer stores its merge table, so a trained vocabulary is a portable artifact.`,

  `Alpha has a context engine. The context length of the model is configured, the tokenizer truncates or pads to fit that window, and the model refuses sequences longer than its context length instead of silently wrapping around.`,

  `Alpha embeddings come from the model itself. A text is tokenised, passed through the transformer, and the final hidden states are pooled into one vector. The vector is normalised, so similarity is a dot product. An untrained model produces weak embeddings, and Alpha labels that honestly.`,

  `Retrieval augmented generation in Alpha has a fixed pipeline. Documents are ingested, parsed, split into chunks with overlap, embedded, and stored in the vector store. A query is embedded with the same model, the closest chunks are retrieved, the context is assembled with source references, and the language model writes the answer from that context.`,

  `Alpha memory is not a chatbot transcript. Memory has scopes: conversation memory lives for one exchange, session memory lives for one workspace session, and long term memory persists. Nothing is written to long term memory without approval, and every memory can be listed, scored, and deleted.`,

  `An Alpha agent is a loop, not a prompt. The agent plans, selects a tool, executes it, verifies the result, and decides whether the task is finished. The loop is bounded by a step budget, every step is recorded, and no tool runs without a permission check.`,

  `Alpha tools are registered with a name, a description, a schema, and a required permission. Tool discovery reads the registry. Tool execution validates arguments against the schema, checks permission, runs the handler, and normalises the result. Failures are values, not exceptions that vanish.`,

  `Alpha speaks the model context protocol shape. A tool descriptor with a name, a description, and an input schema can be registered like any Alpha tool, so an external MCP server can be attached without making it the centre of the system.`,

  `The Alpha vector store owns its vectors. Collections are namespaces, every vector carries metadata, and search is a cosine similarity scan with a deterministic tie break. Insert, update, delete, and persistence are explicit operations.`,

  `Alpha automation runs workflows. A workflow has a trigger, a list of conditions, and a list of actions. Jobs are queued, retried with backoff, and every execution is written to history with its duration and outcome.`,

  `Alpha security is a policy engine, not a slogan. Each actor has a role, each role has permissions, each agent has a scope, and each tool declares the permission it needs. Audit records are append only. Rate limits are token buckets.`,

  `Alpha observability records traces, spans, metrics, and structured logs. Every request receives a trace id, every module reports its work as a span, and the metrics registry keeps counters and histograms for latency and token usage.`,

  `A model has stages. Architecture is the shape of the code. Untrained means the weights are random initialisation. Trained means a training run produced a checkpoint. Fine tuned means a trained model continued learning on a narrower corpus. Production means a human promoted it. Alpha never calls an untrained model finished.`,

  `Training Alpha is resumable. A checkpoint stores the weights, the optimiser moments, the random generator position, the step number, and the metrics measured so far. Loading a checkpoint continues the same run instead of starting a new one.`,

  `Alpha does not hide a provider behind a curtain. There is no key for a hosted model in this repository, no fallback to a third party when the local model is weak, and no demo string pretending to be a prediction. The model says what it learned and nothing more.`,

  `The architecture is small on purpose. A small model trains quickly, fails visibly, and can be inspected line by line. When the code is right the same interfaces accept a larger configuration: more layers, more width, a longer context, and a larger corpus.`,
];

export const ALPHA_SEED_CORPUS: AlphaDataset = createDataset({
  id: "dataset_alpha_seed_v1",
  name: "alpha-seed",
  version: "1.0.0",
  description:
    "Original prose describing the Alpha architecture, written for this repository and used as the default training corpus.",
  license: "CC0-1.0 (authored for this repository)",
  source: "authored for Alpha",
  documents,
});

/** Convenience for demos that want a smaller slice of the same corpus. */
export function seedCorpusSlice(count: number): AlphaDataset {
  return {
    ...ALPHA_SEED_CORPUS,
    id: `${ALPHA_SEED_CORPUS.id}_slice${count}`,
    documents: ALPHA_SEED_CORPUS.documents.slice(0, Math.max(1, count)),
  };
}
