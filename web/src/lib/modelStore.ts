import { CodedError, asCoded } from './errors'
import type { Key, Params } from '../i18n'
import { adaboostPredict, argmax, parseAdaBoostSpec } from './adaboost'
import { TRACE_OUTPUTS, adaboostTrace, type Trace } from './trace'
import { createOnnxPredictor } from './onnx'
import { loadOrt } from './ortLoader'
import type { Manifest, ModelId, Prediction } from './types'
import { MODEL_IDS } from './types'

export type ModelStatus =
  | { state: 'idle' }
  | { state: 'loading' }
  | { state: 'ready'; loadMs: number }
  | { state: 'error'; key: Key; params?: Params }

export type StatusMap = Partial<Record<ModelId, ModelStatus>>

interface Runner {
  run(pixels: Float32Array): Promise<Float64Array>
  trace(pixels: Float32Array): Promise<Trace | null>
}

const FETCH_TIMEOUT_MS = 60_000

async function fetchResponse(url: string, label: string): Promise<Response> {
  const ctl = new AbortController()
  const timer = setTimeout(() => ctl.abort(), FETCH_TIMEOUT_MS)
  try {
    const res = await fetch(url, { signal: ctl.signal })
    if (!res.ok) throw new CodedError('err.http', { code: res.status, file: label })
    return res
  } catch (err) {
    if (err instanceof CodedError) throw err
    throw new CodedError('err.download', { file: label })
  } finally {
    clearTimeout(timer)
  }
}

/** Lazily loads the four classifiers and serialises inference (one ONNX session call at a time). */
export class ModelStore {
  private statuses: StatusMap = {}
  private runners = new Map<ModelId, Runner>()
  private inflight = new Map<ModelId, Promise<void>>()
  private listeners = new Set<() => void>()
  private queue: Promise<unknown> = Promise.resolve()

  constructor(
    readonly manifest: Manifest,
    private readonly baseUrl: string,
  ) {
    for (const id of this.ids()) this.statuses[id] = { state: 'idle' }
  }

  /** Models that the manifest describes and that have a browser implementation. */
  ids(): ModelId[] {
    return MODEL_IDS.filter((id) => {
      const m = this.manifest.models[id]
      return m && (m.format === 'onnx' || m.format === 'json')
    })
  }

  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }

  getSnapshot = (): StatusMap => this.statuses

  private set(id: ModelId, status: ModelStatus) {
    this.statuses = { ...this.statuses, [id]: status }
    this.listeners.forEach((fn) => fn())
  }

  load(id: ModelId): Promise<void> {
    const status = this.statuses[id]
    if (status?.state === 'ready') return Promise.resolve()
    const existing = this.inflight.get(id)
    if (existing) return existing
    const job = this.doLoad(id).finally(() => this.inflight.delete(id))
    this.inflight.set(id, job)
    return job
  }

  private async doLoad(id: ModelId): Promise<void> {
    const entry = this.manifest.models[id]
    if (!entry) return
    this.set(id, { state: 'loading' })
    const t0 = performance.now()
    try {
      const url = `${this.baseUrl}/${entry.file}`
      const label = entry.file.split('/').pop() ?? entry.file
      let runner: Runner
      if (entry.format === 'json') {
        const res = await fetchResponse(url, label)
        let spec: ReturnType<typeof parseAdaBoostSpec>
        try {
          spec = parseAdaBoostSpec(await res.json())
        } catch {
          throw new CodedError('err.format', { file: label })
        }
        runner = { run: async (pixels) => adaboostPredict(spec, pixels), trace: async (pixels) => adaboostTrace(spec, pixels) }
      } else {
        const [res, ort] = await Promise.all([fetchResponse(url, label), loadOrt()])
        const bytes = new Uint8Array(await res.arrayBuffer())
        const predictor = await createOnnxPredictor(ort, bytes, entry.input ?? 'pixels', entry.output ?? 'probabilities')
        runner = {
          run: (pixels) => predictor.run(pixels, 1),
          trace: async (pixels) => {
            const names = TRACE_OUTPUTS[id]
            const o = names ? await predictor.trace(pixels, names) : null
            if (!o) return null
            if (id === 'mlp') return { kind: 'mlp', hidden: [o.a0, o.a1], logits: o.logits }
            if (id === 'softmax') return { kind: 'softmax', x: o.x, weights: o.weights, bias: o.bias, logits: o.logits }
            return { kind: 'naive_bayes', x: o.x, mean: o.class_mean, variance: o.class_var, logPrior: o.log_prior, logits: o.logits }
          },
        }
      }
      this.runners.set(id, runner)
      this.set(id, { state: 'ready', loadMs: performance.now() - t0 })
    } catch (err) {
      const coded = asCoded(err)
      this.set(id, { state: 'error', key: coded.key, params: coded.params })
    }
  }

  /** Intermediate values for the "look inside" panel; null when the loaded file has no trace outputs. */
  trace(id: ModelId, pixels: Float32Array): Promise<Trace | null> {
    const runner = this.runners.get(id)
    if (!runner) return Promise.reject(new CodedError('err.notLoaded'))
    const job = this.queue.then(() => runner.trace(pixels))
    this.queue = job.catch(() => undefined)
    return job
  }

  predict(id: ModelId, pixels: Float32Array): Promise<Prediction> {
    const runner = this.runners.get(id)
    if (!runner) return Promise.reject(new CodedError('err.notLoaded'))
    const job = this.queue.then(async () => {
      const t0 = performance.now()
      const probs = await runner.run(pixels)
      return { probs, top: argmax(probs), ms: performance.now() - t0 }
    })
    this.queue = job.catch(() => undefined)
    return job
  }
}
