/**
 * Model registry and executor: loads the files named in the manifest (verifying their SHA-256),
 * runs ONNX Runtime Web with exactly the extra outputs the trace contract declares, and returns an
 * `Execution` that carries everything an explorer view needs for ONE input and ONE model.
 *
 * Stale-result protection: every Execution is stamped with the model key (bundle / dataset / model)
 * and a fingerprint of the input pixels. Views must ask `matches(exec, ...)` before drawing and show
 * a pending state otherwise, so a view can never show results of a previous drawing or model.
 */
import type { Key, Params } from '../i18n'
import { CodedError, asCoded } from '../lib/errors'
import { adaboostRun, parseAdaBoostBinary, type AdaBoostEnsemble, type AdaBoostRun } from './adaboost'
import { fingerprint, sha256Hex } from './bytes'
import { ContractError, parseReplay, type BundleManifest, type ModelEntry, type ModelId, type ReplayData, type Trace } from './contract'
import { loadParams, type ParamSet } from './params'

export const INPUT_DIM = 784

/** Minimal slice of onnxruntime-web used here (also satisfied by the Node build in tests). */
export interface OrtLike {
  InferenceSession: { create(bytes: Uint8Array, opts?: unknown): Promise<SessionLike> }
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown
}
interface SessionLike {
  inputNames: readonly string[]
  outputNames: readonly string[]
  run(feeds: Record<string, unknown>, names?: string[]): Promise<Record<string, { data: unknown; dims: readonly number[] }>>
  release(): Promise<void>
}

export interface Execution {
  seq: number
  key: string
  dataset: string
  model: ModelId
  /** fingerprint of the 784 input pixels */
  inputId: string
  pixels: Float32Array
  /** softmax probabilities, or normalised vote shares for AdaBoost (NOT calibrated) */
  probs: Float64Array
  top: number
  ms: number
  /** extra ONNX outputs by name (empty for AdaBoost) */
  outputs: Record<string, Float32Array>
  /** per-round routes and cumulative votes (AdaBoost only) */
  ada?: AdaBoostRun
}

export interface LoadedModel {
  key: string
  dataset: string
  model: ModelId
  entry: ModelEntry
  trace: Trace
  params: ParamSet | null
  ensemble: AdaBoostEnsemble | null
  loadMs: number
  execute(pixels: Float32Array): Promise<Execution>
  dispose(): Promise<void>
}

export type ModelStatus =
  | { state: 'idle' }
  | { state: 'loading' }
  | { state: 'ready'; loadMs: number }
  | { state: 'error'; key: Key; params?: Params }

export interface RegistryOptions {
  fetchImpl?: typeof fetch
  ortLoader: () => Promise<OrtLike>
  verifyHashes?: boolean
}

export const modelKey = (m: BundleManifest, dataset: string, model: ModelId) => `${m.bundle_id}/${dataset}/${model}`

/** True only when `exec` was produced by this exact model for exactly these pixels. */
export function matches(exec: Execution | null | undefined, manifest: BundleManifest, dataset: string, model: ModelId, pixels: Float32Array | null): exec is Execution {
  return !!exec && !!pixels && exec.key === modelKey(manifest, dataset, model) && exec.inputId === fingerprint(pixels)
}

const FETCH_TIMEOUT_MS = 90_000

export class ModelRegistry {
  private statuses: Record<string, ModelStatus> = {}
  private loaded = new Map<string, LoadedModel>()
  private inflight = new Map<string, Promise<LoadedModel>>()
  private listeners = new Set<() => void>()
  private queue: Promise<unknown> = Promise.resolve()
  private seq = 0

  constructor(
    readonly manifest: BundleManifest,
    private readonly baseUrl: string,
    private readonly opts: RegistryOptions,
  ) {}

  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }

  getSnapshot = () => this.statuses

  status(dataset: string, model: ModelId): ModelStatus {
    return this.statuses[`${dataset}/${model}`] ?? { state: 'idle' }
  }

  private setStatus(id: string, s: ModelStatus) {
    this.statuses = { ...this.statuses, [id]: s }
    this.listeners.forEach((fn) => fn())
  }

  get(dataset: string, model: ModelId): LoadedModel | undefined {
    return this.loaded.get(`${dataset}/${model}`)
  }

  private async fetchBuf(path: string, expectedSha: string | null): Promise<ArrayBuffer> {
    const url = `${this.baseUrl}/${path}`
    const label = path.split('/').pop() ?? path
    const ctl = new AbortController()
    const timer = setTimeout(() => ctl.abort(), FETCH_TIMEOUT_MS)
    let buf: ArrayBuffer
    try {
      const res = await (this.opts.fetchImpl ?? fetch)(url, { signal: ctl.signal })
      if (!res.ok) throw new CodedError('err.http', { code: res.status, file: label })
      buf = await res.arrayBuffer()
    } catch (err) {
      if (err instanceof CodedError) throw err
      throw new CodedError('err.download', { file: label })
    } finally {
      clearTimeout(timer)
    }
    if (expectedSha && this.opts.verifyHashes !== false && (await sha256Hex(buf)) !== expectedSha) {
      throw new CodedError('err.hash', { file: label })
    }
    return buf
  }

  load(dataset: string, model: ModelId): Promise<LoadedModel> {
    const id = `${dataset}/${model}`
    const done = this.loaded.get(id)
    if (done) return Promise.resolve(done)
    const running = this.inflight.get(id)
    if (running) return running
    const job = this.doLoad(dataset, model).finally(() => this.inflight.delete(id))
    this.inflight.set(id, job)
    return job
  }

  private async doLoad(dataset: string, model: ModelId): Promise<LoadedModel> {
    const id = `${dataset}/${model}`
    const entry = this.manifest.datasets[dataset]?.models[model]
    if (!entry) throw new CodedError('err.generic')
    this.setStatus(id, { state: 'loading' })
    const t0 = performance.now()
    try {
      const key = modelKey(this.manifest, dataset, model)
      const { trace } = entry
      const label = entry.file.split('/').pop() ?? entry.file
      const base = { key, dataset, model, entry, trace }
      let lm: LoadedModel
      if (entry.format === 'binary') {
        const buf = await this.fetchBuf(`${dataset}/${entry.file}`, entry.sha256)
        let ensemble: AdaBoostEnsemble
        try {
          ensemble = parseAdaBoostBinary(buf)
        } catch {
          throw new CodedError('err.format', { file: label })
        }
        const layer = trace.layers.find((l) => l.op === 'tree_ensemble')
        if (!layer || layer.n_estimators !== ensemble.trees.length || layer.node_counts?.some((n, i) => n !== ensemble.nodeCounts[i])) {
          throw new CodedError('err.contract', { file: label })
        }
        lm = {
          ...base,
          params: null,
          ensemble,
          loadMs: 0,
          execute: async (pixels) => {
            const t = performance.now()
            const ada = adaboostRun(ensemble, pixels)
            return this.stamp(base, pixels, ada.shares, ada.top, performance.now() - t, {}, ada)
          },
          dispose: async () => undefined,
        }
      } else {
        const [onnx, ort] = await Promise.all([this.fetchBuf(`${dataset}/${entry.file}`, entry.sha256), this.opts.ortLoader()])
        let params: ParamSet | null = null
        if (trace.params) {
          const pbuf = await this.fetchBuf(`${dataset}/${trace.params.file}`, trace.params.sha256)
          params = await loadParams(trace, pbuf, false)
        }
        const session = await ort.InferenceSession.create(new Uint8Array(onnx), { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
        const wanted = trace.runtime_outputs.map((r) => r.name)
        if (!session.inputNames.includes(entry.input) || !['probabilities', ...wanted].every((n) => session.outputNames.includes(n))) {
          await session.release()
          throw new CodedError('err.signature')
        }
        const names = ['probabilities', ...wanted]
        lm = {
          ...base,
          params,
          ensemble: null,
          loadMs: 0,
          execute: async (pixels) => {
            const t = performance.now()
            const out = await session.run({ [entry.input]: new ort.Tensor('float32', pixels, [1, INPUT_DIM]) }, names)
            const ms = performance.now() - t
            const probs = Float64Array.from(out.probabilities.data as Float32Array)
            if (probs.length !== trace.output.classes.length) throw new CodedError('err.size')
            const outputs: Record<string, Float32Array> = {}
            for (const ro of trace.runtime_outputs) {
              const o = out[ro.name]
              const want = ro.shape.reduce((a, b) => a * b, 1)
              if ((o.data as Float32Array).length !== want) throw new ContractError(`output ${ro.name} has ${(o.data as Float32Array).length} values, expected ${want}`, key)
              outputs[ro.name] = Float32Array.from(o.data as Float32Array)
            }
            let top = 0
            for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i
            return this.stamp(base, pixels, probs, top, ms, outputs)
          },
          dispose: () => session.release(),
        }
      }
      lm.loadMs = performance.now() - t0
      this.loaded.set(id, lm)
      this.setStatus(id, { state: 'ready', loadMs: lm.loadMs })
      return lm
    } catch (err) {
      const coded = err instanceof ContractError ? new CodedError('err.contract', { file: err.where }) : asCoded(err)
      this.setStatus(id, { state: 'error', key: coded.key, params: coded.params })
      throw coded
    }
  }

  private stamp(
    base: { key: string; dataset: string; model: ModelId },
    pixels: Float32Array,
    probs: Float64Array,
    top: number,
    ms: number,
    outputs: Record<string, Float32Array>,
    ada?: AdaBoostRun,
  ): Execution {
    return { seq: ++this.seq, ...base, inputId: fingerprint(pixels), pixels: Float32Array.from(pixels), probs, top, ms, outputs, ada }
  }

  /** The recorded training run of a model (hash-checked), or null when it has none. */
  async loadReplay(dataset: string, model: ModelId): Promise<ReplayData | null> {
    const ref = this.manifest.datasets[dataset]?.models[model]?.replay
    if (!ref) return null
    const buf = await this.fetchBuf(`${dataset}/${ref.file}`, ref.sha256)
    try {
      return parseReplay(JSON.parse(new TextDecoder().decode(buf)), { dataset, model })
    } catch {
      throw new CodedError('err.contract', { file: ref.file.split('/').pop() ?? ref.file })
    }
  }

  /** Runs the model on one image (784 values in [0, 1]); calls are serialised. */
  async execute(dataset: string, model: ModelId, pixels: Float32Array): Promise<Execution> {
    const lm = this.loaded.get(`${dataset}/${model}`) ?? (await this.load(dataset, model))
    const job = this.queue.then(() => lm.execute(pixels))
    this.queue = job.catch(() => undefined)
    return job
  }

  async dispose() {
    await Promise.all([...this.loaded.values()].map((m) => m.dispose()))
    this.loaded.clear()
  }
}
