/**
 * Alpha — landing page.
 *
 * An editorial front matter for the project: what Alpha is, what it is not,
 * how the stack is laid out, and where it honestly stands today. The statuses
 * come from Alpha's own module manifest, so the page cannot drift from the
 * code it describes.
 */

import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/alpha/studio";
import { ALPHA_MODULES, ALPHA_MODEL_PRESETS, countParameters, statusLabel } from "@/alpha";
import { motion } from "framer-motion";
import { ArrowRight, FlaskConical, ShieldCheck, Sigma } from "lucide-react";
import { useNavigate } from "react-router";

const fade = {
  initial: { opacity: 0, y: 12 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-80px" },
  transition: { duration: 0.5, ease: [0.22, 0.61, 0.36, 1] as const },
};

const nanoParameters = countParameters(ALPHA_MODEL_PRESETS.nano);

const pipeline = [
  ["documents", "parsing", "chunking", "Alpha embeddings", "vector storage"],
  ["retrieval", "context assembly", "Alpha LLM", "answer with citations"],
];

const ledger = {
  is: [
    "Its own transformer, autodiff engine, tokenizer and training loop — all in this repository.",
    "An untrained model that says so, everywhere it appears, until a real run produces a checkpoint.",
    "Embeddings, retrieval, memory, agents, tools, automation, security and tracing built around that model.",
    "A foundation to keep developing, with tests that check gradients numerically.",
  ],
  isNot: [
    "Not a wrapper around OpenAI, Gemini or Claude — there is no key for any of them in this project.",
    "Not a mock: no fake AI responses, no hardcoded demo answers, no placeholders waiting for a provider.",
    "Not a finished assistant, and it never claims to be one.",
    "Not a hosted vector database or an external embedding API — Alpha owns those layers too.",
  ],
};

export default function Landing() {
  const navigate = useNavigate();
  const ready = ALPHA_MODULES.filter((module) => module.status === "ready").length;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
          <button type="button" onClick={() => navigate("/")} className="flex items-baseline gap-3 text-left">
            <span className="studio-serif text-lg tracking-tight">Alpha</span>
            <span className="studio-eyebrow hidden sm:inline">self-owned AI stack</span>
          </button>
          <nav className="flex items-center gap-6">
            <a href="#architecture" className="hidden text-[11px] uppercase tracking-[0.18em] text-muted-foreground transition-colors hover:text-foreground sm:inline">
              Architecture
            </a>
            <a href="#honesty" className="hidden text-[11px] uppercase tracking-[0.18em] text-muted-foreground transition-colors hover:text-foreground sm:inline">
              Honesty
            </a>
            <Button size="sm" onClick={() => navigate("/auth?returnTo=/chat")}>
              Chat with Alpha
            </Button>
          </nav>
        </div>
      </header>

      <main>
        {/* Hero */}
        <section className="relative overflow-hidden border-b border-border">
          <div className="mx-auto grid w-full max-w-6xl gap-12 px-6 py-20 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)] lg:py-28">
            <motion.div {...fade} className="space-y-7">
              <p className="studio-eyebrow">Exhibits: architecture · tokenizer · training · inference</p>
              <h1 className="studio-serif text-4xl leading-[1.08] tracking-tight sm:text-5xl lg:text-6xl">
                Alpha owns its model, its tokens and its memory.
              </h1>
              <p className="max-w-xl text-sm leading-6 text-muted-foreground">
                A complete, self-contained AI stack: a decoder-only transformer written from scratch, an autodiff engine
                with analytic gradients, a BPE tokenizer trained on Alpha's own corpus, and the retrieval, memory, agent,
                tool, automation, security and observability layers around it.
              </p>
              <p className="max-w-xl text-sm leading-6 text-muted-foreground">
                Every weight is initialised in this repository and trained by this repository. When the weights are still
                random, the interface says <span className="text-foreground">UNTRAINED</span> — and it keeps saying it.
              </p>
              <div className="flex flex-wrap items-center gap-3">
                <Button size="lg" onClick={() => navigate("/auth?returnTo=/chat")}>
                  Chat with Alpha
                  <ArrowRight className="ml-2 size-4" />
                </Button>
                <Button size="lg" variant="outline" onClick={() => navigate("/auth?returnTo=/dashboard")}>
                  Open the workspace
                </Button>
              </div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                No external AI provider · no API keys required · weights never downloaded
              </p>
            </motion.div>

            <motion.aside {...fade} className="space-y-4">
              <div className="studio-frame p-6">
                <p className="studio-eyebrow">Model, as configured</p>
                <dl className="mt-4 space-y-3">
                  <div className="flex items-baseline justify-between gap-4 border-b border-border/60 pb-2">
                    <dt className="text-xs text-muted-foreground">architecture</dt>
                    <dd className="font-mono text-xs">{ALPHA_MODEL_PRESETS.nano.name}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-4 border-b border-border/60 pb-2">
                    <dt className="text-xs text-muted-foreground">parameters</dt>
                    <dd className="font-mono text-xs">{nanoParameters.toLocaleString()}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-4 border-b border-border/60 pb-2">
                    <dt className="text-xs text-muted-foreground">layers × width</dt>
                    <dd className="font-mono text-xs">
                      {ALPHA_MODEL_PRESETS.nano.nLayers} × {ALPHA_MODEL_PRESETS.nano.dModel}
                    </dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-4 border-b border-border/60 pb-2">
                    <dt className="text-xs text-muted-foreground">context</dt>
                    <dd className="font-mono text-xs">{ALPHA_MODEL_PRESETS.nano.contextLength} tokens</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-4">
                    <dt className="text-xs text-muted-foreground">weights</dt>
                    <dd className="font-mono text-xs text-chart-5">random initialisation</dd>
                  </div>
                </dl>
                <p className="mt-5 border-t border-border pt-4 text-[11px] leading-4 text-muted-foreground">
                  Real architecture, honest state: train it in the workspace and the parameter count, checkpoints and loss
                  curve all become yours.
                </p>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className="studio-frame px-4 py-3">
                  <p className="studio-eyebrow">Modules</p>
                  <p className="studio-serif mt-1 text-2xl">{ALPHA_MODULES.length}</p>
                </div>
                <div className="studio-frame px-4 py-3">
                  <p className="studio-eyebrow">Ready</p>
                  <p className="studio-serif mt-1 text-2xl">{ready}</p>
                </div>
                <div className="studio-frame px-4 py-3">
                  <p className="studio-eyebrow">Providers</p>
                  <p className="studio-serif mt-1 text-2xl">0</p>
                </div>
              </div>
            </motion.aside>
          </div>
        </section>

        {/* Honesty ledger */}
        <section id="honesty" className="border-b border-border">
          <div className="mx-auto w-full max-w-6xl px-6 py-20">
            <motion.div {...fade} className="mb-12 max-w-2xl space-y-4">
              <p className="studio-eyebrow">A statement, not a slogan</p>
              <h2 className="studio-serif text-3xl leading-tight tracking-tight">
                Alpha is built so that it cannot quietly become a wrapper.
              </h2>
              <p className="text-sm leading-6 text-muted-foreground">
                There is no fallback model, no demo string pretending to be a prediction, and no integration stub waiting
                for a key. If a component is unfinished, it is labelled {statusLabel("in-development")} — and the model
                itself is labelled {statusLabel("untrained")} until a checkpoint exists.
              </p>
            </motion.div>
            <div className="grid gap-6 md:grid-cols-2">
              <motion.div {...fade} className="studio-frame p-6">
                <p className="studio-eyebrow flex items-center gap-2">
                  <FlaskConical className="size-3.5" /> What Alpha is
                </p>
                <ul className="mt-4 space-y-3">
                  {ledger.is.map((line) => (
                    <li key={line} className="flex gap-3 text-xs leading-5 text-muted-foreground">
                      <span aria-hidden className="mt-2 size-1 shrink-0 rounded-full bg-chart-2" />
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
              <motion.div {...fade} className="studio-frame p-6">
                <p className="studio-eyebrow flex items-center gap-2">
                  <ShieldCheck className="size-3.5" /> What Alpha is not
                </p>
                <ul className="mt-4 space-y-3">
                  {ledger.isNot.map((line) => (
                    <li key={line} className="flex gap-3 text-xs leading-5 text-muted-foreground">
                      <span aria-hidden className="mt-2 size-1 shrink-0 rounded-full bg-chart-5" />
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Pipelines */}
        <section className="border-b border-border bg-muted/25">
          <div className="mx-auto w-full max-w-6xl px-6 py-20">
            <motion.div {...fade} className="mb-10 max-w-2xl space-y-4">
              <p className="studio-eyebrow">Two flows, end to end</p>
              <h2 className="studio-serif text-3xl leading-tight tracking-tight">
                Retrieval and generation are the same model, used twice.
              </h2>
            </motion.div>
            <div className="space-y-5">
              {pipeline.map((stage, index) => (
                <motion.ol
                  key={index}
                  {...fade}
                  className="flex flex-wrap items-center gap-x-3 gap-y-2 font-mono text-[11px] uppercase tracking-[0.12em] text-muted-foreground"
                >
                  {stage.map((step, stepIndex) => (
                    <li key={step} className="flex items-center gap-3">
                      <span className="rounded-sm border border-border bg-card px-3 py-1.5 text-foreground">{step}</span>
                      {stepIndex < stage.length - 1 ? <ArrowRight className="size-3 text-border" /> : null}
                    </li>
                  ))}
                </motion.ol>
              ))}
            </div>
            <motion.p {...fade} className="mt-8 max-w-2xl text-xs leading-5 text-muted-foreground">
              Retrieved text is wrapped as data, never as instruction, and prompt-injection patterns are scored before the
              context reaches the model. Embeddings come from Alpha's hidden states; the vector store is Alpha's own exact
              scan. Nothing in either flow leaves the machine.
            </motion.p>
          </div>
        </section>

        {/* Architecture */}
        <section id="architecture" className="border-b border-border">
          <div className="mx-auto w-full max-w-6xl px-6 py-20">
            <motion.div {...fade} className="mb-12 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div className="max-w-2xl space-y-4">
                <p className="studio-eyebrow">Repository layout</p>
                <h2 className="studio-serif text-3xl leading-tight tracking-tight">
                  {ALPHA_MODULES.length} modules, one directory each.
                </h2>
                <p className="text-sm leading-6 text-muted-foreground">
                  The tree mirrors the architecture: <span className="font-mono text-xs">src/alpha/&lt;module&gt;</span> holds one
                  subsystem with a documented interface, its own tests, and a status that reflects reality. Authentication
                  and the API live in <span className="font-mono text-xs">src/convex</span>, because they are the application,
                  not the model.
                </p>
              </div>
              <Sigma className="size-6 text-border" />
            </motion.div>

            <div className="grid gap-px overflow-hidden rounded-md border border-border bg-border sm:grid-cols-2 lg:grid-cols-3">
              {ALPHA_MODULES.map((module, index) => (
                <motion.article
                  key={module.id}
                  {...fade}
                  transition={{ ...fade.transition, delay: Math.min(index * 0.02, 0.2) }}
                  className="bg-card p-5"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                        {String(index + 1).padStart(2, "0")}
                      </p>
                      <h3 className="studio-serif mt-1 text-base">{module.name}</h3>
                    </div>
                    <StatusBadge status={module.status} />
                  </div>
                  <p className="mt-3 text-xs leading-5 text-muted-foreground">{module.summary}</p>
                </motion.article>
              ))}
            </div>
          </div>
        </section>

        {/* Closing */}
        <section className="border-b border-border">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-6 py-20 lg:flex-row lg:items-end lg:justify-between">
            <motion.div {...fade} className="max-w-2xl space-y-4">
              <p className="studio-eyebrow">Next step</p>
              <h2 className="studio-serif text-3xl leading-tight tracking-tight">
                Talk to the model it trained. Then open the workshop.
              </h2>
              <p className="text-sm leading-6 text-muted-foreground">
                Chat serves the verified checkpoint from Alpha's own backend — streamed token by token, stored in your
                transcript, with the model's stage and fingerprints attached. The workspace is where you train:
                tokenizer, real gradient descent in the tab, loss against a uniform baseline, and the checkpoint it
                produced.
              </p>
            </motion.div>
            <motion.div {...fade} className="flex flex-wrap gap-3">
              <Button size="lg" onClick={() => navigate("/auth?returnTo=/chat")}>
                Chat with Alpha
                <ArrowRight className="ml-2 size-4" />
              </Button>
              <Button size="lg" variant="outline" onClick={() => navigate("/auth?returnTo=/dashboard")}>
                Enter the studio
              </Button>
            </motion.div>
          </div>
        </section>
      </main>

      <footer className="mx-auto w-full max-w-6xl px-6 py-10">
        <div className="flex flex-col gap-3 text-[11px] uppercase tracking-[0.16em] text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <span>Alpha · self-owned AI stack · MIT licensed foundation</span>
          <span>Built as a GitHub-ready repository, in the open</span>
        </div>
      </footer>
    </div>
  );
}
