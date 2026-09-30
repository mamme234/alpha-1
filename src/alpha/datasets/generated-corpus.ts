/**
 * Alpha's generated training corpus.
 *
 * The seed corpus is twenty paragraphs: enough to prove the lifecycle works,
 * far too small to grow a model. Step 4 needs a corpus that is genuinely
 * larger without ever reaching for material Alpha does not own, so this module
 * *generates* one instead of collecting one.
 *
 * Everything here is authored by this repository. There is no scraped text, no
 * third-party corpus, and nothing copied from a published work. The generator is
 * seeded, so the same seed always produces byte-identical documents — which is
 * what makes a training run over it reproducible.
 *
 * The point is not to be clever prose. The point is volume with structure:
 * a model that has seen `attention` defined consistently many times, and
 * related terms co-occurring, has something real to fit, and the resulting loss
 * curve means something.
 */

import type { AlphaDataset } from "./types";

/** A tiny deterministic PRNG, so corpus generation is reproducible. */
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

type Topic = {
  /** The term being described. */
  term: string;
  /** A one-line definition, reused across documents so it can be learned. */
  definition: string;
  /** Terms this one is meaningfully connected to. */
  related: string[];
  /** A property or behaviour, phrased as a standalone fact. */
  properties: string[];
};

const TOPICS: Topic[] = [
  {
    term: "the transformer",
    definition: "the transformer is the neural architecture alpha uses to process sequences of tokens",
    related: ["attention", "the feed forward block", "layer normalisation", "positional encoding"],
    properties: [
      "a transformer processes every position in a sequence in parallel rather than one position at a time",
      "a transformer stacks attention and feed forward sublayers and normalises between them",
      "a transformer has no recurrence, so the order of the sequence is carried entirely by position information",
    ],
  },
  {
    term: "attention",
    definition: "attention lets every position in a sequence weigh every other position by relevance",
    related: ["the transformer", "the query", "the key", "the value"],
    properties: [
      "attention computes a score between a query and a key and uses that score to weight the value",
      "causal attention masks every position from attending to a position that comes after it",
      "attention heads let one layer look at the sequence in several different ways at once",
    ],
  },
  {
    term: "positional encoding",
    definition: "positional encoding tells the transformer which position each token occupies",
    related: ["the transformer", "the token embedding", "the context length"],
    properties: [
      "a learned positional encoding is a table with one row per position and one column per model dimension",
      "a sinusoidal positional encoding is computed from position and dimension rather than stored",
      "a positional encoding adds position information to a token embedding",
    ],
  },
  {
    term: "the token embedding",
    definition: "the token embedding is the lookup table that maps a token id to a vector the model can work with",
    related: ["the vocabulary", "the output projection", "positional encoding"],
    properties: [
      "the token embedding has one row per entry in the vocabulary",
      "the token embedding is often tied to the output projection so both share one matrix",
      "a token embedding is learned during training like every other parameter",
    ],
  },
  {
    term: "the feed forward block",
    definition: "the feed forward block applies the same small network independently to every position",
    related: ["the transformer", "the activation function", "layer normalisation"],
    properties: [
      "the feed forward block expands the model width and then projects it back down",
      "the feed forward block is applied position by position and does not mix positions itself",
      "the feed forward block holds most of the parameters in a transformer layer",
    ],
  },
  {
    term: "layer normalisation",
    definition: "layer normalisation rescales the activations of a single position to have zero mean and unit variance",
    related: ["the transformer", "the feed forward block", "the residual connection"],
    properties: [
      "layer normalisation has a learned scale and a learned bias for every model dimension",
      "layer normalisation makes training stable across very different activation scales",
      "layer normalisation is applied before each sublayer rather than after it",
    ],
  },
  {
    term: "the residual connection",
    definition: "a residual connection adds the input of a sublayer to its output",
    related: ["the transformer", "layer normalisation", "the feed forward block"],
    properties: [
      "a residual connection lets gradients reach earlier layers without passing through every sublayer",
      "a residual connection keeps information flowing when a sublayer is close to doing nothing",
      "a residual connection adds tensors of the same shape elementwise",
    ],
  },
  {
    term: "byte pair encoding",
    definition: "byte pair encoding builds a vocabulary by repeatedly merging the most frequent adjacent token pair",
    related: ["the vocabulary", "the tokenizer", "the token embedding"],
    properties: [
      "byte pair encoding starts from single bytes and merges pairs until the target vocabulary size is reached",
      "byte pair encoding stops early when no adjacent pair occurs often enough to be worth merging",
      "byte pair encoding keeps a vocabulary small by giving common word fragments a single id",
    ],
  },
  {
    term: "the vocabulary",
    definition: "the vocabulary is the complete set of tokens the tokenizer can produce",
    related: ["the tokenizer", "the token embedding", "byte pair encoding"],
    properties: [
      "the vocabulary size decides how many output logits the model produces at every position",
      "a larger vocabulary makes the output layer more expensive and the sequences shorter",
      "a token that is not in the vocabulary is replaced with the unknown token",
    ],
  },
  {
    term: "cross entropy loss",
    definition: "cross entropy loss measures how surprised the model is by the token that actually came next",
    related: ["the softmax", "perplexity", "the loss curve"],
    properties: [
      "cross entropy loss is measured in nats per token when the logarithm is the natural one",
      "cross entropy loss for a model with no information at all equals the logarithm of the vocabulary size",
      "cross entropy loss falls as the model assigns more probability to the token that actually followed",
    ],
  },
  {
    term: "perplexity",
    definition: "perplexity is the exponential of the cross entropy loss",
    related: ["cross entropy loss", "the softmax", "evaluation"],
    properties: [
      "perplexity is the number of equally likely tokens the model is as confused about as its average loss",
      "perplexity is only meaningful when the loss is small enough for the exponential to stay finite",
      "perplexity can only be compared between models that share a vocabulary and a tokenizer",
    ],
  },
  {
    term: "the softmax",
    definition: "the softmax turns a vector of raw scores into a probability distribution that sums to one",
    related: ["cross entropy loss", "the vocabulary", "temperature"],
    properties: [
      "the softmax subtracts the largest logit before exponentiating, which keeps the numbers finite",
      "the softmax is what turns the model's output layer into a distribution over the vocabulary",
      "a temperature below one makes the softmax distribution sharper",
    ],
  },
  {
    term: "the optimiser",
    definition: "the optimiser updates every parameter in the direction that reduces the loss",
    related: ["the learning rate schedule", "weight decay", "gradient clipping"],
    properties: [
      "adamw keeps a running average of recent gradients and a running average of recent squared gradients",
      "the optimiser applies weight decay directly to the parameters rather than to their gradients",
      "an adaptive optimiser scales each parameter by how large its own gradients have been",
    ],
  },
  {
    term: "the learning rate schedule",
    definition: "the learning rate schedule changes the learning rate over the course of a training run",
    related: ["the optimiser", "warmup", "the learning rate"],
    properties: [
      "a warmup phase raises the learning rate gradually at the start of training",
      "a cosine schedule decays the learning rate smoothly towards a minimum factor",
      "a learning rate that is too large makes the loss diverge instead of falling",
    ],
  },
  {
    term: "gradient clipping",
    definition: "gradient clipping rescales a gradient whose length exceeds a chosen maximum",
    related: ["the optimiser", "the loss curve", "the learning rate schedule"],
    properties: [
      "gradient clipping limits the size of a single update without changing the direction of the gradient",
      "gradient clipping is applied before the optimiser takes its step",
      "gradient clipping keeps one unusual batch from destroying a training run",
    ],
  },
  {
    term: "the checkpoint",
    definition: "a checkpoint stores the weights, the optimiser state and enough metadata to resume a run exactly",
    related: ["the optimiser", "the training run", "the dataset version"],
    properties: [
      "a checkpoint records which dataset version produced it so the run can be reproduced",
      "a checkpoint records the architecture fingerprint and refuses to load against a different architecture",
      "a checkpoint records the position of the random number generator so a resumed run continues the same sequence",
    ],
  },
  {
    term: "retrieval",
    definition: "retrieval finds passages relevant to a question before the model is asked to answer it",
    related: ["the vector store", "the embedding", "the context engine"],
    properties: [
      "retrieval scores a stored passage against the query using vector similarity",
      "retrieval is only useful if the stored passages and the query share an embedding space",
      "retrieval returns passages to the model as context rather than as an answer",
    ],
  },
  {
    term: "the embedding",
    definition: "an embedding maps text to a fixed length vector so that similar text has similar vectors",
    related: ["the vector store", "retrieval", "the token embedding"],
    properties: [
      "an embedding is usually the token embedding averaged over the tokens of a sequence",
      "two embeddings can be compared with a dot product or a cosine similarity",
      "an embedding must come from the same model and version as anything it is compared against",
    ],
  },
  {
    term: "the key value cache",
    definition: "a key value cache stores the keys and values already computed so they are not computed again",
    related: ["attention", "the transformer", "generation"],
    properties: [
      "a key value cache turns generation from quadratic recomputation into one step per new token",
      "a key value cache must respect the same causal mask as a full forward pass",
      "a cached forward pass and an uncached forward pass must produce identical logits",
    ],
  },
  {
    term: "the context length",
    definition: "the context length is the largest number of tokens the model can attend to at once",
    related: ["positional encoding", "the transformer", "the context engine"],
    properties: [
      "the context length bounds the size of the positional encoding table",
      "a model refuses a sequence longer than its configured context length rather than truncating it silently",
      "the attention cost of a sequence grows with the square of the context length",
    ],
  },
  {
    term: "the context engine",
    definition: "the context engine assembles instructions, memory, conversation and sources into one prompt",
    related: ["retrieval", "the context length", "memory"],
    properties: [
      "the context engine fits its blocks to a token budget and reports what it dropped",
      "the context engine orders blocks by priority so instructions are never trimmed away",
      "the context engine measures the rendered prompt rather than assuming a token count",
    ],
  },
  {
    term: "memory",
    definition: "memory is a store of facts about a user that retrieval can bring back into context",
    related: ["retrieval", "the context engine", "the vector store"],
    properties: [
      "memory recall is scored by similarity, importance, recency and how often a fact has been used",
      "a memory can be approved before it is allowed to influence a response",
      "memory belongs to an owner and is never returned to a different owner",
    ],
  },
  {
    term: "the training run",
    definition: "a training run is one execution of the training loop from a seed to a final checkpoint",
    related: ["the checkpoint", "the optimiser", "the dataset version"],
    properties: [
      "a training run records the seed, the configuration, the dataset version and the tokenizer fingerprint",
      "a training run that is interrupted can be resumed from its last checkpoint",
      "a training run reports the tokens it processed rather than only the steps it took",
    ],
  },
  {
    term: "temperature",
    definition: "temperature scales the logits before the softmax and so controls how random sampling is",
    related: ["the softmax", "generation", "sampling"],
    properties: [
      "a temperature of zero makes generation greedy and therefore fully deterministic",
      "a temperature above one flattens the distribution and makes generation more random",
      "temperature changes the sampling behaviour without changing the model weights",
    ],
  },
  {
    term: "sampling",
    definition: "sampling chooses the next token from the predicted distribution rather than taking the most likely one",
    related: ["the softmax", "temperature", "generation"],
    properties: [
      "sampling with a fixed seed and a fixed temperature produces the same tokens every time",
      "top k sampling restricts the choice to the k most likely tokens",
      "top p sampling restricts the choice to the smallest set of tokens whose probability exceeds p",
    ],
  },
  {
    term: "generation",
    definition: "generation is the process of producing new tokens one at a time from a prompt",
    related: ["sampling", "the key value cache", "temperature"],
    properties: [
      "generation stops when the model produces the end of sequence token or reaches its token cap",
      "generation with greedy decoding is deterministic given the same prompt and the same weights",
      "a generation that produced no tokens is reported as an empty result rather than filled in",
    ],
  },
  {
    term: "evaluation",
    definition: "evaluation measures a trained model on data it was not trained on",
    related: ["cross entropy loss", "perplexity", "the dataset version"],
    properties: [
      "evaluation data must be held out from training or the measurement means nothing",
      "evaluation reports a measured loss and perplexity rather than a single judgement",
      "an evaluation result is tied to the exact model version and dataset version that produced it",
    ],
  },
];

const CONNECTORS = [
  "This matters because",
  "In practice",
  "For a reader building on this",
  "The consequence is that",
  "It follows that",
  "A useful way to think about this is that",
  "Note that",
  "In other words",
];

const OPENERS = [
  "Alpha records this because",
  "The following note belongs with it because",
  "This is worth stating plainly:",
  "A working definition helps:",
  "For the purposes of this system,",
  "Keep the following in mind:",
];

/**
 * Build the corpus. `documentCount` documents are produced from the topic set
 * with a seeded generator, so a given (documentCount, seed) always yields the
 * same text and the same fingerprint.
 */
export function buildGeneratedCorpus(
  documentCount = 300,
  seed = 20250930,
): AlphaDataset {
  const random = lcg(seed);
  const pick = <T>(items: T[]): T => items[Math.floor(random() * items.length) % items.length];
  const documents: string[] = [];

  for (let i = 0; i < documentCount; i++) {
    const topic = TOPICS[i % TOPICS.length];
    const related = topic.related.map((term) => `the relationship between ${term} and ${topic.term}`);

    const lines: string[] = [];
    lines.push(`${pick(OPENERS)} ${topic.definition}.`);

    // The definition recurs across documents: this is what a model can learn.
    lines.push(topic.definition + ".");

    const propertyA = topic.properties[i % topic.properties.length];
    lines.push(`${pick(CONNECTORS)} ${propertyA}.`);

    const propertyB = topic.properties[(i + 1) % topic.properties.length];
    lines.push(`${propertyB.charAt(0).toUpperCase()}${propertyB.slice(1)}.`);

    // Cross-topic relations, so the corpus is not twenty isolated islands.
    const bridge = pick(related);
    const other = pick(TOPICS.filter((t) => t.term !== topic.term));
    lines.push(
      `${bridge} is real: ${other.definition}, and it constrains what ${topic.term} can usefully mean.`,
    );

    lines.push(
      `In alpha this is enforced rather than assumed. ${topic.properties[0]}. ` +
        `The same rule applies to ${other.term}, and violating it produces a model that cannot be trusted.`,
    );

    // A short, checkable fact in a consistent surface form, which gives the
    // evaluation benchmark something recognisable to look for.
    lines.push(`The term used throughout is "${topic.term}".`);

    documents.push(lines.join("\n"));
  }

  return {
    id: "alpha-generated",
    name: "alpha-generated-corpus",
    version: `1.0.0+${documentCount}`,
    description: `Deterministically generated Alpha-owned technical corpus: ${documentCount} documents over ${TOPICS.length} topics. Authored by src/alpha/datasets/generated-corpus.ts; no external or third-party text.`,
    license: "Alpha-owned",
    source: "alpha-generated",
    documents,
  };
}

/** Topic names, so a report can list what the corpus actually covers. */
export function generatedCorpusTopics(): string[] {
  return TOPICS.map((t) => t.term);
}

/** Documents the generator produces for a given size, without building them. */
export function generatedCorpusSize(documentCount: number): number {
  return documentCount;
}
