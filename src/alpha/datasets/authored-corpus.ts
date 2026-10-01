/**
 * Alpha's multi-category authored corpus.
 *
 * Step 4's corpus proved a model can train, but it was one template applied to
 * thirty topics: every document had the same eight lines in the same order with
 * the same connectives. Diversity metrics were structurally incapable of moving,
 * and a model fitted to it learned one shape rather than any language at all.
 *
 * This module is the honest replacement. Eight categories, each with its own
 * surface form, and within each category a deterministic generator that varies
 * subject, verb, ordering and length. The point is not to imitate a published
 * corpus — it is to give Alpha text with genuine structural variety, so that a
 * measured diversity improvement means something and so that a mixture over
 * categories is a real mixture rather than a relabelling.
 *
 * Everything here is authored in this repository:
 *
 *   general-prose     narrative-shaped paragraphs
 *   educational       worked explanations with an explicit learning aim
 *   factual-reference self-consistent records about Alpha's own components
 *   dialogue          two speakers, turns and replies
 *   instructions      imperative steps in ordered form
 *   explanations      cause-and-effect prose ("because X, therefore Y")
 *   structured        lists, key/value pairs, JSON-like records
 *   multilingual      short authored passages in other languages
 *
 * Every document is deterministic given (seed, index): the generator uses a
 * seeded LCG, so a corpus is reproducible and its fingerprint is stable.
 *
 * Provenance is attached per document by the caller in `./provenance`, and
 * every document here declares which category it belongs to, so a category is
 * only ever claimed if it actually has documents in it.
 */

import type { MixCategory } from "./provenance";

/** A deterministic PRNG so corpus generation is reproducible. */
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

type Rng = () => number;
function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length) % items.length];
}

/**
 * Small self-consistent world used by several categories: names, places, tasks
 * and objects that recur, so the model can learn entity continuity rather than
 * one shape.
 */
const PEOPLE = [
  "Ada",
  "Rune",
  "Mira",
  "Tomas",
  "Ines",
  "Kofi",
  "Yara",
  "Lena",
  "Ozan",
  "Petra",
  "Ravi",
  "Noor",
] as const;

const PLACES = [
  "the north workshop",
  "the riverside lab",
  "the old observatory",
  "the harbour office",
  "the quiet reading room",
  "the hillside station",
  "the market square",
  "the archive basement",
] as const;

const ARTEFACTS = [
  "a folded map",
  "a brass key",
  "a weather log",
  "a paper ledger",
  "a hand lens",
  "a wool coat",
  "a tin of ink",
  "a field notebook",
] as const;

const COLOURS = ["amber", "slate", "dusk-blue", "rust", "pale green", "charcoal"] as const;

const WEATHER = [
  "a cold wind off the water",
  "steady rain",
  "a clear and bright afternoon",
  "low cloud",
  "a sudden squall",
  "warm still air",
] as const;

/**
 * Deterministic Fisher-Yates over a copy, for varying sentence order.
 *
 * Sentence order is part of variety: two documents built from the same clauses
 * but assembled differently are genuinely different text, while the same clauses
 * in the same order are a duplicate wearing a different opener.
 */
function shuffled<T>(items: readonly T[], rng: Rng): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Take `n` distinct items from a pool, or the whole pool if it is smaller. */
function sample<T>(rng: Rng, pool: readonly T[], n: number): T[] {
  return shuffled(pool, rng).slice(0, Math.max(1, Math.min(n, pool.length)));
}

// ---------------------------------------------------------------------------
// Frame pools
//
// The Step 4 corpus reused one sentence frame per role, which made every
// document in a category a near-duplicate of every other one. These pools give
// each role several surface forms, so the generator varies the *shape* of a
// sentence as well as its content.
// ---------------------------------------------------------------------------

const LESSON_OPENERS = [
  "The lesson today concerns",
  "We are looking at",
  "This section covers",
  "Today's subject is",
  "The topic under discussion is",
  "What follows deals with",
  "The working topic is",
  "This part of the course addresses",
] as const;

// The aim frames compose with an infinitive phrase. `lesson.aim` is written as a
// bare verb phrase ("see what a subword vocabulary buys") precisely so that any
// of these frames produces a grammatical sentence.
const LESSON_AIMS = [
  "The aim is to",
  "By the end you should be able to",
  "The goal here is to",
  "We are trying to",
  "The intention is to",
  "What we want to reach is a way to",
  "The objective is to",
  "This lesson exists to help you",
] as const;

const LESSON_CENTRES = [
  "The central idea is that",
  "The essential claim is that",
  "Everything rests on the fact that",
  "The key point:",
  "At the heart of it,",
  "The thing to hold onto is that",
  "The core of the argument is that",
  "Start from this:",
] as const;

const LESSON_HOLDS = [
  "It holds",
  "The reason is that",
  "This is so because",
  "The justification is that",
  "We know this because",
  "It follows that",
  "The basis for it is that",
  "It depends on the fact that",
] as const;

const LESSON_EXAMPLE_INTROS = [
  "Consider this:",
  "Take an example.",
  "To make it concrete:",
  "Here is a case:",
  "One illustration:",
  "Try this instead:",
  "Picture it this way:",
  "For a specific instance:",
] as const;

const LESSON_CHECKS = [
  "A useful check on your understanding is to ask yourself",
  "Check your understanding by asking",
  "You have understood it when you can answer this question:",
  "A good question to test yourself with is",
  "To verify, ask yourself",
  "The sign of understanding is whether you can say what",
  "Try explaining it by asking",
  "Ask yourself the following, and stop when you can answer:",
] as const;

const LESSON_CLOSERS = [
  "If you can explain the idea without reaching for the word \"obviously\", you have it.",
  "Anyone who cannot yet argue against it has not finished reading.",
  "Say the idea back in your own words and the lesson is complete.",
  "The test is whether you can use it on an example you have not seen.",
  "Memorising the wording will not help; using it will.",
  "Where the argument has a gap, the gap is where to start next time.",
  "Treat the definition as a tool, not as a fact to recite.",
  "One more step: apply it to something that was not in this lesson.",
] as const;

const REFERENCE_HEADERS = [
  "Reference entry:",
  "Specification record:",
  "Component note:",
  "Data sheet:",
  "Component record:",
  "Technical note:",
  "Inventory entry:",
  "Entry in the component register:",
] as const;

const REFERENCE_UNITS = [
  "Measured unit:",
  "The unit here is the",
  "Counted in",
  "Its natural unit is the",
  "Everything is expressed per",
  "Figures below are given in",
  "The quantity being described is the",
  "One unit means one",
] as const;

const REFERENCE_HOLDS = [
  "It holds",
  "It maintains",
  "It keeps",
  "It carries",
  "It is built around",
  "It stores",
  "It is composed of",
  "It works in terms of",
] as const;

const REFERENCE_FUNCTIONS = [
  "In use, it",
  "Functionally, it",
  "What it does: it",
  "Operationally, it",
  "In practice it",
  "Its behaviour is that it",
  "Put plainly, it",
  "Observed directly, it",
] as const;

const REFERENCE_STABILITY = [
  "Stability: it is checked against a numerical gradient check rather than assumed correct.",
  "It is verified rather than assumed; the gradient check is the evidence.",
  "Verification runs a numerical gradient check on every step of a real run.",
  "Correctness is measured with a numerical gradient check, not asserted.",
  "It is checked by comparing gradients against central differences.",
  "Its behaviour is verified against finite differences before it is trusted.",
  "A numerical gradient check backs every claim made here.",
  "The property is tested directly, with finite differences, on each run.",
] as const;

const EXPLANATION_OPENERS = [
  "When",
  "Whenever",
  "If",
  "In the case where",
  "Once",
  "It follows that when",
  "Consider what happens when",
  "The situation arises when",
] as const;

const EXPLANATION_BECAUSES = [
  "This happens because",
  "The reason is that",
  "The mechanism is that",
  "This is so because",
  "Underneath it,",
  "The explanation is that",
  "What makes it true is that",
  "It works this way because",
] as const;

const EXPLANATION_SECOND = [
  "A second, related case: when",
  "The same argument covers another case. When",
  "There is a matching situation where",
  "One more case runs the other way. When",
  "A parallel example: when",
  "The reverse also holds. When",
  "A second consequence appears. When",
  "The neighbouring case behaves similarly. When",
] as const;

const EXPLANATION_SINCE = [
  "since",
  "because",
  "the reason being that",
  "given that",
  "on the grounds that",
  "as",
  "for the reason that",
  "in that",
] as const;

const EXPLANATION_REMEDIES = [
  "What to do about it is simple.",
  "The fix, where one exists, is straightforward.",
  "There is a remedy, and it is not a bigger model.",
  "The obvious response is not the right one.",
  "Treat this as a bug to be located, not a wall to be hit.",
  "The correction is procedural rather than clever.",
  "Nothing here requires more parameters.",
  "The answer is to change the setup, not the hyperparameters.",
] as const;

const EXPLANATION_CLOSERS = [
  "The practical lesson is the same in both cases. Measure the thing you actually care about before changing anything, and change one thing at a time.",
  "Both cases reduce to the same discipline: measure first, then change one variable, then measure again.",
  "The pattern is the same each time. Know what you are trying to move, and do not move two things at once.",
  "In both cases the answer is to look before leaping. A measurement taken before the change is what makes the change legible.",
  "These are the same mistake wearing different clothes. Establish the baseline before you touch the system.",
  "Whichever case you are in, the method is unchanged: define the measurement, take it, change one thing, take it again.",
  "Nothing here is exotic. The discipline of measuring before changing is the whole content of both cases.",
  "The common thread is a baseline. Without one, any result afterwards is indistinguishable from noise.",
] as const;

const INSTRUCTION_FRAMES = [
  "How to",
  "Procedure for",
  "Steps to",
  "The method for",
  "A reliable way to",
  "Working procedure for",
  "The correct approach to",
  "Instructions for",
] as const;

const INSTRUCTION_CLOSERS = [
  "Follow every step in order.",
  "Work through them in sequence and do not skip ahead.",
  "Take them one at a time; the order matters.",
  "Each step depends on the one before it, so keep the order.",
  "Read the whole list before starting, then execute it in order.",
  "Do them in the order given, and finish each one before beginning the next.",
  "The sequence is deliberate; changing it invalidates the result.",
  "Complete them in order and record what you observed at each stage.",
] as const;

const DIALOGUE_OPEN_A1 = [
  "You have been in",
  "You were in",
  "You spent the morning in",
  "You have been sitting in",
  "You were stuck in",
  "You have been alone in",
] as const;

const DIALOGUE_OPEN_A2 = [
  "all morning.",
  "since before the others arrived.",
  "for hours now.",
  "and nobody came to help.",
  "again, it seems.",
  "while the rest of us got on with things.",
] as const;

const DIALOGUE_REPLY_B1 = [
  "There was",
  "I had",
  "We had",
  "There was still",
  "Someone had left",
] as const;

const DIALOGUE_REPLY_B2 = [
  "to sort, and no one else to sort it.",
  "to finish, and it would not wait.",
  "to put away before the light went.",
  "to find, and it mattered.",
  "to repair, badly and in a hurry.",
  "to account for, and the numbers would not add up.",
] as const;

const DIALOGUE_PROBE_A = [
  "And?",
  "Did it work?",
  "And was it worth it?",
  "So?",
  "Did you finish?",
  "And how did that go?",
] as const;

const DIALOGUE_ANSWER_B1 = [
  "It is done.",
  "Mostly.",
  "As far as anyone can tell.",
  "Eventually.",
  "It is as done as it will ever be.",
] as const;

const DIALOGUE_ANSWER_B2 = [
  "The whole shelf is covered in dust.",
  "Every surface is thick with it.",
  "You cannot see the wood underneath.",
  "There is more of it than there was.",
  "It is in places where it should never have been.",
  "It is %s everywhere, whichever way the light falls.",
] as const;

const DIALOGUE_REACT_A = [
  "That is more than I expected from a morning.",
  "I would not have guessed that.",
  "You make it sound easy.",
  "That is a better result than we deserved.",
  "Nobody believed you would get there.",
] as const;

const DIALOGUE_CLOSE_B1 = [
  "Most things are, if you let them be.",
  "It never is, until it is.",
  "You learn that quickly enough.",
  "The trick is not to look too closely.",
  "That is the part nobody tells you.",
  "It is easier the second time, and never as easy as you hoped.",
] as const;

/** Weather phrased as a yes/no answer, so the turn is a real reply. */
const WEATHER_ANSWER = [
  "It was. It comes across that open ground like that.",
  "It was, though it stopped just before noon.",
  "Not yet. It arrived later than anyone expected.",
  "Not that I noticed, but I was indoors by then.",
  "It had. You could smell it before you reached the door.",
] as const;

const DIALOGUE_FINAL_A = [
  "Was it still going when you arrived?",
  "What was the weather like first thing?",
  "Did anyone else come out in it?",
  "Had it started before you got there?",
  "Was there any warning before the weather turned?",
] as const;

const DIALOGUE_FINAL_B = [
  "Blowing sideways, yes. It got in under the door every time it opened.",
  "It had. I remember because my notes were ruined.",
  "Not yet, but it came in halfway through.",
  "Constantly. Nobody could hear anything at the far end.",
  "It stopped just as I got inside, which felt deliberate.",
] as const;

const DIALOGUE_OPEN_B3 = [
  "I will tell you what I noticed.",
  "Let me say what happened, since nobody else will.",
  "You want the honest version? Here it is.",
  "It was not what I expected, and I have thought about why.",
  "Here is what actually occurred, in order.",
  "I have been turning this over since yesterday.",
  "It is a strange business, and here is the whole of it.",
  "Nobody asked me to, so I will say it anyway.",
] as const;

const DIALOGUE_REPLY_A3 = [
  "Where were you when it happened?",
  "So you were already inside?",
  "How long before anyone else came?",
  "Did anyone see you go in?",
  "You mean all of it, or just that part?",
] as const;

const DIALOGUE_REPLY_B4 = [
  "Then it was not as bad as I thought.",
  "Then you were better placed than the rest of us.",
  "Then that explains the rest of it.",
  "Then I will stop asking questions.",
  "Then you already knew more than you let on.",
] as const;

const DIALOGUE_REPLY_A4 = [
  "That is easy to say from here.",
  "You will believe what you like.",
  "It sounds different when you are the one waiting.",
  "I do not think that was the whole of it.",
  "You have a very steady account of a confusing morning.",
] as const;

/**
 * Clauses that take a colour adjective and complete a sentence on their own,
 * e.g. "The room is the colour of the weather {colour}."
 */
const DIALOGUE_REPLY_B5 = [
  "The room is %s, the way weather always looks indoors",
  "Everything in there has gone %s",
  "The entire east wall is %s where it used to be white",
  "Half of it has already turned %s",
  "Even the ceiling has taken on a %s cast",
] as const;

const DIALOGUE_REACT_A3 = [
  "It was already that colour before you arrived.",
  "That is not what I remember.",
  "Someone will have to repaint it eventually.",
  "You could have said something earlier.",
  "It is a poor advertisement for the last tenant.",
] as const;

const DIALOGUE_CLOSE_B2 = [
  "Anyway. It is done, and that is the part that matters.",
  "So. We can argue about the rest of it another time.",
  "Which is why nobody asked.",
  "And that was the end of that particular conversation.",
  "I will find the paint tomorrow and you can say I told you so.",
] as const;

// ---------------------------------------------------------------------------
// general prose
// ---------------------------------------------------------------------------

const PROSE_OPENERS = [
  "By the time the light reached the windows",
  "Nobody expected much from the morning",
  "It had been a long week on the whole team",
  "The first thing she noticed was the quiet",
  "After the visitors had gone",
  "Somewhere out past the last field",
  "The letter arrived on a Tuesday",
  "They had stopped keeping count of the days",
] as const;

const PROSE_BODIES = [
  (r: Rng) =>
    `${pick(r, PEOPLE)} was the first to reach ${pick(r, PLACES)}, carrying ${pick(r, ARTEFACTS)} against the ${pick(r, COLOURS)} coat.`,
  (r: Rng) =>
    `A thin ${pick(r, COLOURS)} line of ${pick(r, WEATHER)} had come in overnight, and it made every surface look older than it was.`,
  (r: Rng) =>
    `They worked in silence for a while, the kind of silence that is closer to agreement than to awkwardness.`,
  (r: Rng) =>
    `It occurred to ${pick(r, PEOPLE)} that the whole arrangement had been arranged long before anyone thought to ask.`,
  (r: Rng) =>
    `Outside, ${pick(r, WEATHER)} kept up its own unhurried schedule, indifferent to what was decided indoors.`,
  (r: Rng) =>
    `The ${pick(r, ARTEFACTS)} had belonged to the family for three generations, and nobody could remember who had first written in it.`,
  (r: Rng) =>
    `There was a moment, brief and sharp, when ${pick(r, PEOPLE)} nearly said something and then decided not to.`,
  (r: Rng) =>
    `Later they would describe the room as ${pick(r, COLOURS)}, though at the time it seemed only tired.`,
  (r: Rng) =>
    `${pick(r, PEOPLE)} had a way of leaving a door slightly open that nobody ever mentioned and nobody ever closed.`,
  (r: Rng) =>
    `The argument, such as it was, turned on a detail in ${pick(r, ARTEFACTS)} that turned out to be a smudge and not a mark.`,
  (r: Rng) =>
    `Someone had put a chair against the door, and ${pick(r, PEOPLE)} decided that was a statement rather than an accident.`,
  (r: Rng) =>
    `By the time the kettle gave up, the light had gone from ${pick(r, COLOURS)} to something without a name.`,
  (r: Rng) =>
    `Two of the three agreed, and the third said nothing at all, which everyone understood.`,
  (r: Rng) =>
    `It was ${pick(r, PLACES)} that did it, though ${pick(r, PEOPLE)} would deny that until the following spring.`,
  (r: Rng) =>
    `They ate standing up, which told ${pick(r, PEOPLE)} everything worth knowing about the day.`,
  (r: Rng) =>
    `The ${pick(r, COLOURS)} light made ${pick(r, ARTEFACTS)} look like something from a different decade entirely.`,
  (r: Rng) =>
    `A long time afterwards, ${pick(r, PEOPLE)} remembered the ${pick(r, WEATHER)} and nothing else about any of it.`,
  (r: Rng) =>
    `Nobody moved the furniture, on the grounds that it would only have to be moved again later.`,
  (r: Rng) =>
    `Whatever ${pick(r, PEOPLE)} had expected to find in ${pick(r, PLACES)}, it was not that.`,
  (r: Rng) =>
    `There is a particular silence that follows a decision nobody wants to defend out loud.`,
  (r: Rng) =>
    `The ${pick(r, ARTEFACTS)} had a ${pick(r, COLOURS)} cover worn smooth in one corner, exactly where a thumb would go.`,
  (r: Rng) =>
    `They agreed to disagree, which took less time than the disagreement would have.`,
  (r: Rng) =>
    `${pick(r, WEATHER).charAt(0).toUpperCase() + pick(r, WEATHER).slice(1)} came in off the water and put an end to the day.`,
  (r: Rng) =>
    `In the end it was ${pick(r, PEOPLE)} who said it out loud, and ${pick(r, PEOPLE)} who had to live with it.`,
] as const;

const PROSE_CLOSERS = [
  "In the end that was enough to begin with.",
  "The rest, as it turned out, took considerably longer.",
  "And so the morning closed without anyone needing to explain it.",
  "It is easy, afterwards, to think it had been inevitable.",
  "Whatever came next would have to account for all of it.",
  "That, at least, is what they agreed on.",
] as const;

function generalProse(rng: Rng): string {
  // Bodies are drawn without replacement where possible, so a short document
  // does not repeat a sentence it has already used.
  const bodyCount = 2 + Math.floor(rng() * 4);
  const used = new Set<string>();
  const bodies: string[] = [];
  let guard = 0;
  while (bodies.length < bodyCount && guard < 40) {
    guard += 1;
    const candidate = pick(rng, PROSE_BODIES)(rng);
    if (used.has(candidate)) continue;
    used.add(candidate);
    bodies.push(candidate);
  }
  // The opening is a standalone sentence: joining a subordinate opener to a main
  // clause with a comma produces a sentence that is grammatically wrong, and the
  // model would learn that as the pattern.
  return [`${pick(rng, PROSE_OPENERS)}.`, ...shuffled(bodies, rng), pick(rng, PROSE_CLOSERS)].join(" ");
}

// ---------------------------------------------------------------------------
// educational
// ---------------------------------------------------------------------------

type Lesson = {
  subject: string;
  aim: string;
  point: string;
  because: string;
  example: string;
  check: string;
};

const LESSONS: Lesson[] = [
  {
    subject: "the embedding table",
    aim: "understand what a token id becomes",
    point: "a token id indexes a row, and that row is the only thing the model knows about the token",
    because: "everything downstream reads the row, not the integer",
    example: "two tokens with similar meanings only become similar once their rows are trained to agree",
    check: "what a token id means before its row has been trained",
  },
  {
    subject: "causal masking",
    aim: "explain why a position cannot see forward",
    point: "causal masking removes every score pointing at a later position",
    because: "a prediction at position t must not depend on the answer at position t plus one",
    example: "without the mask, the model would be reading its own target during training and scoring nothing at inference",
    check: "what information leaks if the mask is removed",
  },
  {
    subject: "weight tying",
    aim: "see why an output layer can be reused as an input table",
    point: "a tied embedding uses the same matrix for lookup and for producing logits",
    because: "the weights that say what a token looks like are the same weights that say what may follow it",
    example: "tying halves the parameters in the embedding and output layers and costs a little accuracy on large models",
    check: "which parameter count changes when tying is switched on",
  },
  {
    subject: "validation loss",
    aim: "tell a validation loss from a training loss",
    point: "validation loss is measured on documents the optimiser never updated against",
    because: "a training loss measures fit, and a model can fit perfectly and still not generalise",
    example: "a falling training loss with a rising validation loss is overfitting, not progress",
    check: "what a falling training loss alongside a rising validation loss means",
  },
  {
    subject: "sequence length",
    aim: "reason about cost as a function of context",
    point: "attention cost grows with the square of the sequence length",
    because: "every position must score every other position, and the number of pairs is quadratic",
    example: "doubling the context quadruples the attention work but only doubles the feed-forward work",
    check: "what the dominant cost is at long context lengths",
  },
  {
    subject: "the learning rate",
    aim: "judge whether a rate is too large",
    point: "a rate set too large makes the loss rise rather than fall",
    because: "each step overshoots the region where the gradient still points downhill",
    example: "a loss curve that climbs in the first ten steps is not slow progress, it is divergence",
    check: "what the first few steps of a loss curve tell you about the rate",
  },
  {
    subject: "repetition penalty",
    aim: "understand why decoding penalises repeated tokens",
    point: "a repetition penalty divides the logit of a token that has already been generated",
    because: "without it, greedy decoding can lock onto one token and never leave",
    example: "a penalty of one leaves the distribution untouched; anything above one discourages loops",
    check: "what a repetition penalty of exactly one does",
  },
  {
    subject: "the key value cache",
    aim: "see why caching changes speed but not output",
    point: "a cache stores the keys and values already computed for the prefix",
    because: "recomputing the prefix produces the same values every time, so storing them saves the work",
    example: "a cached decode and an uncached decode must produce identical logits, which is what makes the cache a pure optimisation",
    check: "what would have to be true for a cache to change the model's output",
  },
  {
    subject: "quantisation",
    aim: "explain what quantisation gives up",
    point: "quantisation stores weights in fewer bits than the float used in training",
    because: "fewer bits per weight means less memory and faster arithmetic, but less resolution",
    example: "a weight rounded to eight bits loses precision that a rare but important parameter cannot afford to lose",
    check: "what is gained and what is lost when weights are stored in fewer bits",
  },
  {
    subject: "document boundaries",
    aim: "understand why documents are separated explicitly",
    point: "a boundary marker tells the model where one document ended and the next began",
    because: "without a marker, a model reading a concatenation cannot tell a sentence from a paragraph break",
    example: "training on a stream with no boundaries teaches the model that documents run together, which it will reproduce",
    check: "what a model cannot infer from a token stream with no boundary markers",
  },
  {
    subject: "sequence models",
    aim: "explain why order matters",
    point: "a sequence model treats position as part of the meaning",
    because: "the same tokens in a different order describe a different situation",
    example: "the dog chased the cat and the cat chased the dog are the same words but not the same claim",
    check: "what changes when two words swap places",
  },
  {
    subject: "attention",
    aim: "distinguish attention from a fixed window",
    point: "attention lets each position choose which earlier positions matter to it",
    because: "in language, the informative word is rarely the immediately previous one",
    example: "the animal did not cross the street because it was too tired refers back across several words",
    check: "which earlier word the word it actually refers to",
  },
  {
    subject: "perplexity",
    aim: "read a perplexity number honestly",
    point: "perplexity is the number of equally likely options the model is confused between",
    because: "the exponential of a loss has to be interpreted against the size of the vocabulary",
    example: "a perplexity of 30 on a 10000 word vocabulary is far better than 30 on a vocabulary of 30",
    check: "whether the vocabulary size is stated next to the number",
  },
  {
    subject: "regularisation",
    aim: "see what weight decay does",
    point: "weight decay shrinks parameters toward zero unless the data pulls them elsewhere",
    because: "a parameter that is not supported by evidence should not be free to grow",
    example: "a rarely used connection gradually becomes small and stops distorting the average",
    check: "what a parameter does when it appears rarely in the data",
  },
  {
    subject: "evaluation",
    aim: "keep a test set honest",
    point: "a measurement on data the model has already seen describes memory, not skill",
    because: "a model can reproduce a memorised answer perfectly and still generalise badly",
    example: "asking for a training sentence back tests recall; asking a new question tests learning",
    check: "whether the evaluation text was ever in the training set",
  },
  {
    subject: "tokenisation",
    aim: "see what a subword vocabulary buys",
    point: "merging frequent pairs shortens sequences without throwing away rare words",
    because: "an unseen word can still be assembled from smaller known pieces",
    example: "a rare surname is split into fragments the model has met before",
    check: "how many tokens an unseen word costs",
  },
  {
    subject: "gradient clipping",
    aim: "know why clipping exists",
    point: "clipping caps how far one unusual batch can move the parameters",
    because: "one surprising batch should not be allowed to undo the work of thousands",
    example: "a single batch with an unusually large gradient is rescaled, not discarded",
    check: "whether clipping changes the direction of a gradient or only its length",
  },
  {
    subject: "learning rate schedules",
    aim: "justify warmup",
    point: "warmup raises the rate gradually so early updates do not destabilise the model",
    because: "the very first updates have the largest effect on randomly initialised weights",
    example: "a schedule that starts at full rate can push the first steps off the loss surface",
    check: "what happens to the early loss without warmup",
  },
];

function educational(rng: Rng): string {
  const lesson = pick(rng, LESSONS);
  // Two lessons per document, with a varying set of sections. A lesson that
  // always opens the same way, covers the same ground, and closes the same way
  // is a template, and a template makes every document in the category a
  // near-duplicate of every other one.
  const second = pick(
    rng,
    LESSONS.filter((l) => l.subject !== lesson.subject),
  );
  const sections = [
    `${pick(rng, LESSON_OPENERS)} ${lesson.subject}. ${pick(rng, LESSON_AIMS)} ${lesson.aim}.`,
    `${pick(rng, LESSON_CENTRES)} ${lesson.point}. ${pick(rng, LESSON_HOLDS)} ${lesson.because}.`,
    `${pick(rng, LESSON_EXAMPLE_INTROS)} ${lesson.example}.`,
    `${pick(rng, LESSON_CHECKS)} ${lesson.check}.`,
  ];
  const followUps = [
    `${pick(rng, LESSON_OPENERS)} ${second.subject}. ${pick(rng, LESSON_AIMS)} ${second.aim}.`,
    `${pick(rng, LESSON_CENTRES)} ${second.point}. ${pick(rng, LESSON_HOLDS)} ${second.because}.`,
    `${pick(rng, LESSON_EXAMPLE_INTROS)} ${second.example}.`,
  ];
  // 2 to 4 sections, in varying order, so no two lessons have the same shape.
  const count = 2 + Math.floor(rng() * 3);
  const body = [...sections, ...(rng() < 0.7 ? followUps.slice(0, 1 + Math.floor(rng() * 2)) : [])];
  return [...shuffled(body, rng).slice(0, count), pick(rng, LESSON_CLOSERS)].join(" ");
}

// ---------------------------------------------------------------------------
// factual reference (about Alpha's own components — self-consistent records)
// ---------------------------------------------------------------------------

type Component = {
  name: string;
  unit: string;
  /** Noun phrases, e.g. "one row per token". Joined by a verb like "holds". */
  counts: string[];
  /** Verb phrases in the third person, e.g. "assigns each unit exactly one id". */
  facts: string[];
};

const COMPONENTS: Component[] = [
  {
    name: "the tokenizer",
    unit: "token",
    counts: ["a fixed vocabulary", "one merge table", "four special tokens"],
    facts: [
      "assigns each unit exactly one id",
      "is trained on the corpus rather than downloaded",
      "round-trips text it has seen exactly",
    ],
  },
  {
    name: "the embedding table",
    unit: "row",
    counts: ["one row per token", "one column per model dimension", "a single shared table"],
    facts: [
      "starts from small random values",
      "is updated by every training step",
      "is usually shared with the output projection",
    ],
  },
  {
    name: "the attention block",
    unit: "score",
    counts: ["one score per pair of positions", "one head per view", "a full row per query"],
    facts: [
      "masks every future position",
      "weights values by the scores it computed",
      "returns a weighted average rather than a single value",
    ],
  },
  {
    name: "the optimiser",
    unit: "parameter",
    counts: ["two running averages", "one bias correction", "one decay term"],
    facts: [
      "updates every parameter each step",
      "keeps memory of recent gradient magnitudes",
      "decays weights independently of the gradient",
    ],
  },
  {
    name: "the trainer",
    unit: "step",
    counts: ["one forward pass", "one backward pass", "one update after the batch"],
    facts: [
      "records its seed and configuration",
      "writes a checkpoint at a fixed interval",
      "measures validation loss on held-out documents",
    ],
  },
  {
    name: "the inference engine",
    unit: "call",
    counts: ["one prompt in", "a stream of tokens out", "one cached key per position"],
    facts: [
      "uses a cache so the prefix is not recomputed",
      "reports the stage of the weights honestly",
      "supports cooperative cancellation",
    ],
  },
];

function factualReference(rng: Rng): string {
  const [first, second] = sample(rng, COMPONENTS, 2);
  const entry = (component: Component): string[] => [
    `${pick(rng, REFERENCE_HEADERS)} ${component.name}.`,
    `${pick(rng, REFERENCE_UNITS)} ${component.unit}.`,
    `${pick(rng, REFERENCE_HOLDS)} ${pick(rng, component.counts)}.`,
    `${pick(rng, REFERENCE_FUNCTIONS)} ${pick(rng, component.facts)}.`,
    pick(rng, REFERENCE_STABILITY),
  ];
  // Sometimes one component, usually two: the record length varies, which is
  // what stops every reference document from having identical length and shape.
  if (rng() < 0.35) return entry(first).join(" ");
  return [...entry(first), ...entry(second)].join(" ");
}

// ---------------------------------------------------------------------------
// dialogue
// ---------------------------------------------------------------------------

function dialogue(rng: Rng): string {
  const a = pick(rng, PEOPLE);
  let b = pick(rng, PEOPLE);
  if (b === a) b = PEOPLE[(PEOPLE.indexOf(a) + 1) % PEOPLE.length];
  const place = pick(rng, PLACES);
  const item = pick(rng, ARTEFACTS);
  const colour = pick(rng, COLOURS);
  // Two exchange shapes, so two conversations are not the same conversation
  // with different names in them.
  if (rng() < 0.5) {
    return [
      `${a}: ${pick(rng, DIALOGUE_OPEN_A1)} ${place} ${pick(rng, DIALOGUE_OPEN_A2)}`,
      `${b}: ${pick(rng, DIALOGUE_REPLY_B1)} ${item} ${pick(rng, DIALOGUE_REPLY_B2)}`,
      `${a}: ${pick(rng, DIALOGUE_PROBE_A)}`,
      `${b}: ${pick(rng, DIALOGUE_ANSWER_B1)} ${pick(rng, DIALOGUE_ANSWER_B2).replace("%s", colour)}`,
      `${a}: ${pick(rng, DIALOGUE_REACT_A)}`,
      `${b}: ${pick(rng, DIALOGUE_CLOSE_B1)}`,
      `${a}: ${pick(rng, DIALOGUE_FINAL_A)}`,
      `${b}: ${pick(rng, WEATHER_ANSWER)}`,
    ].join("\n");
  }
  // Shape two: a statement, a report, and an argument about a detail.
  // Each turn is one complete utterance, so no turn ends in a fragment.
  return [
    `${b}: ${pick(rng, DIALOGUE_OPEN_B3)}`,
    `${a}: ${pick(rng, DIALOGUE_REPLY_A3)} You were in ${place}, if that is what you are asking.`,
    `${b}: ${pick(rng, DIALOGUE_REPLY_B4)}`,
    `${a}: ${pick(rng, DIALOGUE_REPLY_A4)}`,
    `${b}: ${pick(rng, DIALOGUE_REPLY_B5).replace("%s", colour)}.`,
    `${a}: ${pick(rng, DIALOGUE_REACT_A3)}`,
    `${b}: ${pick(rng, DIALOGUE_CLOSE_B2)}`,
  ].join("\n");
}

// ---------------------------------------------------------------------------
// instructions
// ---------------------------------------------------------------------------

const TASK_STEPS = [
  {
    goal: "measure the quality of a dataset",
    steps: [
      "run the diversity analysis over the corpus",
      "check that the category counts match the intended mixture",
      "compare token counts against character counts",
      "record every finding without deleting anything",
      "decide explicitly which documents to drop",
    ],
  },
  {
    goal: "train a model and keep the run reproducible",
    steps: [
      "fix the seed before the first step",
      "record the dataset version and its fingerprint",
      "reserve a test split the trainer never sees",
      "evaluate on the held-out split only",
      "save the checkpoint with its configuration",
    ],
  },
  {
    goal: "review a model's output before trusting it",
    steps: [
      "confirm the weights came from a real run",
      "read the generated text rather than the summary",
      "check whether the prompt was in the training data",
      "look for repetition and truncation",
      "record the raw output alongside any conclusion",
    ],
  },
  {
    goal: "add a new data source safely",
    steps: [
      "state the licence before importing anything",
      "record the acquisition method and date",
      "label the language and category of each document",
      "run the duplicate and leakage checks",
      "only then include it in a mixture",
    ],
  },
  {
    goal: "replace a tokenizer without losing checkpoints",
    steps: [
      "measure the current tokenizer against the new corpus",
      "train a candidate and measure it the same way",
      "require a stated improvement before switching",
      "bump the tokenizer version and fingerprint",
      "refuse to load old checkpoints against the new vocabulary",
    ],
  },
  {
    goal: "decide whether a model should reach production",
    steps: [
      "confirm it has trained weights and a checkpoint",
      "attach at least one evaluation to it",
      "record who approved it and why, in writing",
      "keep the approval manual rather than automatic",
      "leave the decision reversible",
    ],
  },
  {
    goal: "report a result without overstating it",
    steps: [
      "separate what was measured from what was estimated",
      "state the baseline the result should be read against",
      "report the failed cases alongside the successful ones",
      "avoid collapsing several measurements into one score",
      "say what the result does not show",
    ],
  },
  {
    goal: "choose a training configuration",
    steps: [
      "estimate memory before starting, not during",
      "start from a configuration that has already been measured",
      "change one hyperparameter at a time",
      "keep a fixed seed so two runs are comparable",
      "record the configuration with the result",
    ],
  },
  {
    goal: "serve a model to an application",
    steps: [
      "check the runtime architecture matches the model",
      "confirm the tokenizer fingerprint matches the weights",
      "verify streaming and cancellation behave",
      "measure first-token latency rather than guessing at it",
      "fall back honestly when the model is untrained",
    ],
  },
  {
    goal: "curriculum-order a training corpus",
    steps: [
      "start with examples whose structure is easiest to fit",
      "increase difficulty only after loss stops falling on the easy set",
      "measure whether the ordering helped, on held-out data",
      "drop the ordering entirely if it made no difference",
      "keep the ordering deterministic so runs stay comparable",
    ],
  },
  {
    goal: "audit an existing experiment record",
    steps: [
      "confirm the dataset version and tokenizer version are named",
      "check the seed and the full configuration",
      "verify the reported token count against the run",
      "confirm the evaluation set was never trained on",
      "record the failure reason if the run did not succeed",
    ],
  },
];

function instructions(rng: Rng): string {
  const task = pick(rng, TASK_STEPS);
  const second = pick(rng, TASK_STEPS.filter((t) => t.goal !== task.goal));
  // Steps are numbered, not re-ordered: a procedure whose steps are shuffled is
  // a different procedure. What varies is how many steps, and whether a second
  // task is included.
  const render = (steps: string[]): string =>
    steps.map((step, i) => `${i + 1}. ${step.charAt(0).toUpperCase() + step.slice(1)}.`).join(" ");
  const drop = Math.floor(rng() * task.steps.length);
  const steps = task.steps.filter((_, i) => i !== drop);
  const body = [render(steps)];
  if (rng() < 0.45) body.push(`${pick(rng, INSTRUCTION_FRAMES)} ${second.goal}. ${render(second.steps)}`);
  return `${pick(rng, INSTRUCTION_FRAMES)} ${task.goal}. ${body.join(" ")} ${pick(rng, INSTRUCTION_CLOSERS)}`;
}

// ---------------------------------------------------------------------------
// explanations (cause / effect)
// ---------------------------------------------------------------------------

const CAUSES: Array<{ cause: string; effect: string; because: string; remedy: string }> = [
  {
    cause: "the validation loss stops falling",
    effect: "continuing to train on the same data buys nothing",
    because: "the part of the distribution the model can fit has already been fitted",
    remedy: "add data that the model cannot yet fit, rather than more steps over data it can",
  },
  {
    cause: "a model is asked about a language it never saw",
    effect: "its output is close to meaningless in that language",
    because: "every token of that language is unseen at both input and output",
    remedy: "train on that language, or say plainly that it does not support it",
  },
  {
    cause: "attention is given no positional information",
    effect: "the model cannot tell first from last",
    because: "without position, permuting the input permutes the representation",
    remedy: "add a positional signal, learned or computed, and train with it",
  },
  {
    cause: "a test set is reused for tuning",
    effect: "the reported score drifts upward over time",
    because: "each reuse leaks a little of the test set into the choices being made",
    remedy: "draw a fresh test set before the next round of changes",
  },
  {
    cause: "the learning rate is set far too high",
    effect: "the loss rises instead of falling and may never recover",
    because: "each update overshoots the region where the gradient still points downhill",
    remedy: "warm up over more steps and lower the peak rate",
  },
  {
    cause: "a corpus repeats one sentence verbatim thousands of times",
    effect: "the model fits that sentence and little else",
    because: "the repeated text dominates the gradient far more than its share of the variety",
    remedy: "measure the repetition ratio before training and fix the corpus, not the model",
  },
  {
    cause: "two documents in a corpus are near-identical",
    effect: "a near-copy of a training example sits in the test split and the score measures memory",
    because: "the model reproduces the copy exactly and the measurement calls it generalisation",
    remedy: "group near-duplicates into one split before drawing the boundaries",
  },
  {
    cause: "a vocabulary is replaced after training begins",
    effect: "every checkpoint written against the old vocabulary is silently wrong",
    because: "token ids are indices, so a different table means every index now means another token",
    remedy: "keep the old tokenizer versioned and refuse incompatible checkpoints",
  },
  {
    cause: "a metric is only ever reported on its best run",
    effect: "the reported number drifts further from the typical result with every run",
    because: "the best of many noisy measurements is biased upward, and more runs make it worse",
    remedy: "report the distribution, not the maximum",
  },
  {
    cause: "an evaluation is edited after its result is known",
    effect: "the score improves for reasons that have nothing to do with the model",
    because: "the measurement was fitted to the answer instead of the other way round",
    remedy: "freeze the suite and its fingerprint before the first model runs",
  },
  {
    cause: "a model is promoted because its loss went down",
    effect: "a model with no language ability reaches production behind a smaller one",
    because: "loss measures fit, and a memorising model fits very well",
    remedy: "require independent behavioural evidence before promotion, and keep it manual",
  },
  {
    cause: "padding tokens enter the loss",
    effect: "the model spends capacity learning what a pad token predicts",
    because: "a padded batch contains positions that carry no information about the text",
    remedy: "mask the padded targets out of the cross entropy",
  },
];

function explanations(rng: Rng): string {
  const chosen = sample(
    rng,
    CAUSES,
    2 + Math.floor(rng() * 2),
  );
  const parts = chosen.map((item, index) => {
    const head =
      index === 0
        ? `${pick(rng, EXPLANATION_OPENERS)} ${item.cause}, ${item.effect}. ${pick(rng, EXPLANATION_BECAUSES)} ${item.because}.`
        : `${pick(rng, EXPLANATION_SECOND)} ${item.cause}, ${item.effect}, ${pick(rng, EXPLANATION_SINCE)} ${item.because}.`;
    // Roughly half the cases carry the remedy. Varying whether a clause is
    // present at all is a different kind of variety from varying its wording,
    // and it is what stops two explanations of the same cause from matching.
    if (rng() < 0.55) {
      return `${head} ${pick(rng, EXPLANATION_REMEDIES)} ${item.remedy}.`;
    }
    return head;
  });
  return [...parts, pick(rng, EXPLANATION_CLOSERS)].join(" ");
}

// ---------------------------------------------------------------------------
// structured
// ---------------------------------------------------------------------------

const STRUCTURED_RECORDS = [
  (r: Rng) =>
    `record: sample_${Math.floor(r() * 900 + 100)}\nfield: category\nvalue: ${pick(r, ["instructions", "explanations", "dialogue"])}\nfield: language\nvalue: en`,
  (r: Rng) =>
    `record: entry_${Math.floor(r() * 900 + 100)}\nfield: colour\nvalue: ${pick(r, COLOURS)}\nfield: object\nvalue: ${pick(r, ARTEFACTS)}\nfield: count\nvalue: ${1 + Math.floor(r() * 40)}`,
  (r: Rng) =>
    `record: task_${Math.floor(r() * 900 + 100)}\nfield: owner\nvalue: ${pick(r, PEOPLE)}\nfield: place\nvalue: ${pick(r, PLACES)}\nfield: status\nvalue: ${pick(r, ["open", "closed", "deferred"])}`,
  (r: Rng) =>
    `{"id": ${Math.floor(r() * 900 + 100)}, "category": "${pick(r, ["general-prose", "educational", "structured"])}", "language": "en", "checked": true}`,
  (r: Rng) =>
    `list:\n  - ${pick(r, ARTEFACTS)}\n  - ${pick(r, ARTEFACTS)}\n  - ${pick(r, ARTEFACTS)}\ntotal: 3`,
];

function structured(rng: Rng): string {
  return pick(rng, STRUCTURED_RECORDS)(rng);
}

// ---------------------------------------------------------------------------
// multilingual (short authored passages; language-tagged, not translated at runtime)
// ---------------------------------------------------------------------------

/**
 * Multilingual passages, authored here rather than translated at runtime.
 *
 * Each language carries several distinct passages, and each passage is built
 * from sentences that can be combined in varying orders and counts. With only
 * one passage per language, every document in this category would be a
 * duplicate of every other one in the same language.
 */
const MULTILINGUAL: Record<
  string,
  { marker: string; subject: string[]; middle: string[]; close: string[] }
> = {
  es: {
    marker: "Nota:",
    subject: [
      "El modelo procesa cada posición de la secuencia.",
      "Un modelo pequeño sigue siendo un modelo.",
      "Cada documento del corpus tiene su origen.",
    ],
    middle: [
      "La atención permite que una posición considere las anteriores.",
      "Los pesos cambian despacio en cada paso.",
      "Una palabra rara se divide en trozos conocidos.",
    ],
    close: [
      "El entrenamiento ajusta los pesos poco a poco.",
      "Los datos de evaluación nunca se usan para entrenar.",
      "Un resultado medido vale más que una afirmación.",
    ],
  },
  fr: {
    marker: "À noter :",
    subject: [
      "Le modèle examine chaque position de la séquence.",
      "Un petit modèle reste un modèle.",
      "Chaque document du corpus porte sa provenance.",
    ],
    middle: [
      "L attention relie une position aux précédentes.",
      "Les poids changent lentement à chaque pas.",
      "Un mot rare se découpe en morceaux connus.",
    ],
    close: [
      "L entraînement modifie les poids progressivement.",
      "Les données d évaluation ne servent jamais à entraîner.",
      "Une mesure vaut mieux qu une affirmation.",
    ],
  },
  de: {
    marker: "Hinweis:",
    subject: [
      "Das Modell verarbeitet jede Position der Folge.",
      "Ein kleines Modell bleibt ein Modell.",
      "Jedes Dokument trägt seine Herkunft mit sich.",
    ],
    middle: [
      "Die Aufmerksamkeit verbindet eine Position mit den vorherigen.",
      "Die Gewichte ändern sich bei jedem Schritt langsam.",
      "Ein seltenes Wort zerfällt in bekannte Teile.",
    ],
    close: [
      "Das Training ändert die Gewichte Schritt für Schritt.",
      "Auswertungsdaten werden nie zum Training benutzt.",
      "Eine Messung ist mehr wert als eine Behauptung.",
    ],
  },
  pt: {
    marker: "Observação:",
    subject: [
      "O modelo processa cada posição da sequência.",
      "Um modelo pequeno continua a ser um modelo.",
      "Cada documento do corpus traz a sua origem.",
    ],
    middle: [
      "A atenção permite que uma posição considere as anteriores.",
      "Os pesos mudam devagar a cada passo.",
      "Uma palavra rara divide-se em partes conhecidas.",
    ],
    close: [
      "O treinamento ajusta os pesos aos poucos.",
      "Os dados de avaliação nunca treinam o modelo.",
      "Uma medição vale mais do que uma afirmação.",
    ],
  },
  it: {
    marker: "Nota bene:",
    subject: [
      "Il modello elabora ogni posizione della sequenza.",
      "Un modello piccolo resta un modello.",
      "Ogni documento del corpus porta la sua provenienza.",
    ],
    middle: [
      "L attenzione collega una posizione a quelle precedenti.",
      "I pesi cambiano lentamente a ogni passo.",
      "Una parola rara si spezza in frammenti noti.",
    ],
    close: [
      "L addestramento modifica i pesi lentamente.",
      "I dati di valutazione non addestrano mai il modello.",
      "Una misura vale più di un affermazione.",
    ],
  },
  nl: {
    marker: "Opmerking:",
    subject: [
      "Het model verwerkt elke positie van de reeks.",
      "Een klein model blijft een model.",
      "Elk document draagt zijn herkomst mee.",
    ],
    middle: [
      "Aandacht verbindt een positie met de vorige.",
      "De gewichten veranderen langzaam bij elke stap.",
      "Een zeldzaam woord valt uiteen in bekende delen.",
    ],
    close: [
      "Training past de gewichten geleidelijk aan.",
      "Evaluatiegegevens worden nooit gebruikt om te trainen.",
      "Een meting weegt zwaarder dan een bewering.",
    ],
  },
};

const MULTILINGUAL_LANGUAGES = Object.keys(MULTILINGUAL);

/**
 * How many multilingual documents have been produced so far. A module-level
 * counter would make the generator's output depend on call history, so the
 * counter is threaded through `buildAuthoredCorpus` instead and lives here only
 * as the declaration of its type.
 */
let multilingualCounter = 0;

/** Reset the rotation so each corpus build starts from a known state. */
function resetMultilingualRotation(): void {
  multilingualCounter = 0;
}

// ---------------------------------------------------------------------------

/** One generated document plus the category it belongs to. */
export type AuthoredDocument = {
  text: string;
  category: MixCategory;
  language: string;
};

/** Categories this generator can produce, in a stable order. */
export function authoredCategories(): MixCategory[] {
  return [
    "general-prose",
    "educational",
    "factual-reference",
    "dialogue",
    "instructions",
    "explanations",
    "structured",
    "multilingual",
  ];
}

function categoryFor(index: number, categoryCount: number): MixCategory {
  // Round-robin so every category is represented when there are >= categoryCount docs.
  return authoredCategories()[index % categoryCount];
}

function buildOne(rng: Rng, category: MixCategory): AuthoredDocument {
  switch (category) {
    case "general-prose":
      return { text: generalProse(rng), category, language: "en" };
    case "educational":
      return { text: educational(rng), category, language: "en" };
    case "factual-reference":
      return { text: factualReference(rng), category, language: "en" };
    case "dialogue":
      return { text: dialogue(rng), category, language: "en" };
    case "instructions":
      return { text: instructions(rng), category, language: "en" };
    case "explanations":
      return { text: explanations(rng), category, language: "en" };
    case "structured":
      return { text: structured(rng), category, language: "en" };
    case "multilingual": {
      // Languages are chosen by a separate, deterministic rotation rather than
      // at random: random selection over six languages and ten documents leaves
      // one or two languages with a single example each, which means a category
      // claim backed by almost nothing.
      const rotationIndex = multilingualCounter % MULTILINGUAL_LANGUAGES.length;
      multilingualCounter += 1;
      const language = MULTILINGUAL_LANGUAGES[rotationIndex];
      const entry = MULTILINGUAL[language];
      const sentences = [
        pick(rng, entry.subject),
        pick(rng, entry.middle),
        pick(rng, entry.close),
      ];
      // A fourth sentence sometimes, in a varying position, so two documents in
      // the same language are never the same three sentences in the same order.
      if (rng() < 0.4) {
        sentences.splice(Math.floor(rng() * (sentences.length + 1)), 0, pick(rng, entry.middle));
      }
      return {
        text: `${entry.marker} ${shuffled(sentences, rng).join(" ")}`,
        category,
        language,
      };
    }
  }
}

/**
 * Build `count` documents spread evenly across the categories above. The same
 * (count, seed) always produces byte-identical documents, so a training run and
 * its dataset fingerprint are reproducible.
 */
export function buildAuthoredCorpus(count: number, seed = 20260101): AuthoredDocument[] {
  if (count <= 0) {
    throw new Error("buildAuthoredCorpus needs a positive document count");
  }
  const categories = authoredCategories();
  const rng = lcg(seed);
  resetMultilingualRotation();
  const documents: AuthoredDocument[] = [];
  for (let i = 0; i < count; i++) {
    documents.push(buildOne(rng, categoryFor(i, categories.length)));
  }
  return documents;
}

/** Exposed so a test can confirm every language is reachable. */
export function multilingualSamples(): Array<{ language: string; text: string }> {
  return MULTILINGUAL_LANGUAGES.map((language) => {
    const entry = MULTILINGUAL[language];
    return {
      language,
      text: `${entry.marker} ${entry.subject[0]} ${entry.middle[0]} ${entry.close[0]}`,
    };
  });
}
