/**
 * Alpha Inference Service — the interface every caller uses.
 *
 * This is the transport-neutral entry point to Alpha's generation:
 *
 *   generate({ modelId, prompt, generationConfig })
 *   generateStream({ modelId, prompt, generationConfig })
 *   scoreNextTokens({ modelId, context })
 *
 * It holds *handles* (a model, its tokenizer and the honest stage of its
 * weights) rather than talking to a provider. There is no default remote model,
 * no fallback, and an unknown `modelId` fails loudly with the list of models
 * that do exist.
 *
 * The module imports nothing from React, Vite, Convex or the DOM, so the same
 * service runs inside the browser workspace, in a Node script, or in a test.
 */

import { AlphaValidationError } from "../core/errors";
import type { AlphaModelStage } from "../core/types";
import type { AlphaTransformer } from "../model/transformer";
import type { AlphaTokenizer } from "../tokenizer/bpe";
import {
  AlphaInferenceEngine,
  type GenerationResult,
  type GenerationStreamChunk,
  type SamplingConfig,
} from "./engine";

export type AlphaModelHandle = {
  id: string;
  name: string;
  version: string;
  stage: AlphaModelStage;
  model: AlphaTransformer;
  tokenizer: AlphaTokenizer;
  contextLength: number;
  /** Checkpoint the weights came from, when the model has trained. */
  checkpointId: string | null;
  tokenizerFingerprint: string;
  loadedAt: number;
};

export type RegisterModelInput = Omit<AlphaModelHandle, "loadedAt" | "tokenizerFingerprint"> & {
  loadedAt?: number;
  tokenizerFingerprint?: string;
};

export type GenerateRequest = {
  /** Defaults to the most recently registered model. */
  modelId?: string;
  prompt: string;
  generationConfig?: Partial<SamplingConfig>;
};

export type GenerateStreamRequest = GenerateRequest;

export type ScoreRequest = {
  modelId?: string;
  context: number[];
};

export type AlphaModelDescriptor = {
  id: string;
  name: string;
  version: string;
  stage: AlphaModelStage;
  contextLength: number;
  checkpointId: string | null;
  tokenizerFingerprint: string;
  vocabularySize: number;
  parameterCount: number;
  loadedAt: number;
};

export class AlphaInferenceService {
  private readonly handles = new Map<string, AlphaModelHandle>();
  private readonly engines = new Map<string, AlphaInferenceEngine>();
  private defaultId: string | null = null;

  /** Build the stable id for a model + tokenizer pair. */
  static modelId(name: string, version: string, tokenizerFingerprint: string): string {
    return `${name}@${version}+${tokenizerFingerprint}`;
  }

  registerModel(input: RegisterModelInput): AlphaModelHandle {
    if (input.tokenizer.vocabSize > input.model.config.vocabSize) {
      throw new AlphaValidationError(
        "inference",
        `tokenizer vocabulary (${input.tokenizer.vocabSize}) is larger than the model vocabulary (${input.model.config.vocabSize})`,
      );
    }
    const handle: AlphaModelHandle = {
      ...input,
      tokenizerFingerprint: input.tokenizerFingerprint ?? input.tokenizer.fingerprint(),
      loadedAt: input.loadedAt ?? Date.now(),
    };
    this.handles.set(handle.id, handle);
    // A re-registered id must not keep serving the previous weights.
    this.engines.delete(handle.id);
    this.defaultId = handle.id;
    return handle;
  }

  unregisterModel(modelId: string): boolean {
    this.engines.delete(modelId);
    const removed = this.handles.delete(modelId);
    if (this.defaultId === modelId) {
      this.defaultId = this.handles.size > 0 ? [...this.handles.keys()][this.handles.size - 1] : null;
    }
    return removed;
  }

  clear(): void {
    this.handles.clear();
    this.engines.clear();
    this.defaultId = null;
  }

  get defaultModel(): string | null {
    return this.defaultId;
  }

  get size(): number {
    return this.handles.size;
  }

  listModels(): AlphaModelDescriptor[] {
    return [...this.handles.values()].map((handle) => ({
      id: handle.id,
      name: handle.name,
      version: handle.version,
      stage: handle.stage,
      contextLength: handle.contextLength,
      checkpointId: handle.checkpointId,
      tokenizerFingerprint: handle.tokenizerFingerprint,
      vocabularySize: handle.tokenizer.vocabSize,
      parameterCount: handle.model.parameterCount,
      loadedAt: handle.loadedAt,
    }));
  }

  has(modelId: string): boolean {
    return this.handles.has(modelId);
  }

  /** Resolve a handle (and its cached engine) or fail with a useful message. */
  resolve(modelId?: string): { handle: AlphaModelHandle; engine: AlphaInferenceEngine } {
    const id = modelId ?? this.defaultId;
    if (!id) {
      throw new AlphaValidationError(
        "inference",
        "no model is registered with the inference service — initialise the workspace first",
      );
    }
    const handle = this.handles.get(id);
    if (!handle) {
      throw new AlphaValidationError(
        "inference",
        `unknown modelId "${id}"; registered models: ${[...this.handles.keys()].join(", ") || "none"}`,
      );
    }
    let engine = this.engines.get(id);
    if (!engine) {
      engine = new AlphaInferenceEngine({
        model: handle.model,
        tokenizer: handle.tokenizer,
        stage: handle.stage,
        maxContextTokens: handle.contextLength,
      });
      this.engines.set(id, engine);
    }
    return { handle, engine };
  }

  /** One-shot generation from a named model. */
  generate(request: GenerateRequest): GenerationResult {
    const { engine } = this.resolve(request.modelId);
    return engine.generate(request.prompt, request.generationConfig);
  }

  /** Streaming generation from a named model. */
  generateStream(request: GenerateStreamRequest): AsyncGenerator<GenerationStreamChunk, GenerationResult, void> {
    const { engine } = this.resolve(request.modelId);
    return engine.generateStream(request.prompt, request.generationConfig);
  }

  /** Next-token logits for diagnostics. */
  scoreNextTokens(request: ScoreRequest): Float32Array {
    const { engine } = this.resolve(request.modelId);
    return engine.scoreNextTokens(request.context);
  }

  /** The sampling configuration a request would actually use. */
  resolveSampling(modelId: string | undefined, partial: Partial<SamplingConfig> = {}): SamplingConfig {
    const { engine } = this.resolve(modelId);
    return engine.resolveSampling(partial);
  }

  /** Metadata for traces and the UI — never claims more than the handles hold. */
  describe(): {
    registered: number;
    defaultModelId: string | null;
    stages: Partial<Record<AlphaModelStage, number>>;
    models: AlphaModelDescriptor[];
  } {
    const models = this.listModels();
    const stages: Partial<Record<AlphaModelStage, number>> = {};
    for (const model of models) stages[model.stage] = (stages[model.stage] ?? 0) + 1;
    return { registered: models.length, defaultModelId: this.defaultId, stages, models };
  }
}
