/**
 * Pure helpers that turn a model's `trace.layers` into the pieces the MLP / CNN views draw.
 * Nothing here knows how many layers, neurons, channels or kernels a model has: everything is
 * read from the trace, so a deeper network or another dataset is drawn without code changes.
 */
import type { Trace, TraceLayer } from '../../../engine/contract'
import type { StageGeometry } from '../../../engine/reconstruct'

const prod = (s: readonly number[]) => s.reduce((a, b) => a * b, 1)

/** Ops that pass values through unchanged (only the shape is reinterpreted). */
const PASS_THROUGH = new Set(['flatten', 'reshape'])

/**
 * The ONNX output that carries the values of `layerId` for the current input, following
 * shape-only ops back to a layer that has one. null = the raw input pixels (no runtime output).
 */
export function runtimeOf(trace: Trace, layerId: string): string | null {
  const byId = new Map(trace.layers.map((l) => [l.id, l]))
  let cur = byId.get(layerId)
  while (cur) {
    if (cur.runtime) return cur.runtime
    if (!PASS_THROUGH.has(cur.op) || cur.inputs.length === 0) return null
    cur = byId.get(cur.inputs[0])
  }
  return null
}

/** Layers that read `id` as their first input. */
const consumers = (trace: Trace, id: string) => trace.layers.filter((l) => l.inputs[0] === id)

export interface DenseBlock {
  /** position among the dense layers, 0-based */
  index: number
  dense: TraceLayer
  /** ReLU applied to this dense layer's output, if any */
  act: TraceLayer | null
  nIn: number
  nOut: number
  weight: string
  bias: string
  /** id of the layer whose values feed this dense layer, and its runtime output name */
  inputId: string
  inputRuntime: string | null
  /** runtime names of z (pre-activation) and of the activation (null when there is no ReLU) */
  zRuntime: string | null
  aRuntime: string | null
  /** true for the last dense layer when its output goes into the softmax */
  isOutput: boolean
}

export function denseBlocks(trace: Trace): DenseBlock[] {
  const dense = trace.layers.filter((l) => l.op === 'dense' && l.params)
  return dense.map((d, index) => {
    const next = consumers(trace, d.id)
    const act = next.find((l) => l.op === 'relu') ?? null
    const isOutput = !act && (next.some((l) => l.op === 'softmax') || d === trace.layers[trace.layers.length - 1])
    return {
      index,
      dense: d,
      act,
      nIn: d.in_features!,
      nOut: d.out_features!,
      weight: d.params!.weight,
      bias: d.params!.bias,
      inputId: d.inputs[0],
      inputRuntime: runtimeOf(trace, d.inputs[0]),
      zRuntime: d.runtime ?? null,
      aRuntime: act?.runtime ?? null,
      isOutput,
    }
  })
}

export interface ConvStage {
  index: number
  conv: TraceLayer
  act: TraceLayer | null
  pool: TraceLayer | null
  /** [Cin, H, W] of the stage input and the runtime output that holds it */
  inShape: number[]
  inputId: string
  inputRuntime: string | null
  kernel: string
  bias: string
  /** [Cout, Cin, k, k] */
  kernelShape: number[]
  stride: number
  /** conv/pool geometry from the network input up to and including the conv, and up to the pool */
  convStack: StageGeometry[]
  poolStack: StageGeometry[] | null
}

/** Conv / pool layers on the path from the input to `layerId` (inclusive), in input order. */
export function geometryTo(trace: Trace, layerId: string): StageGeometry[] {
  const byId = new Map(trace.layers.map((l) => [l.id, l]))
  const out: StageGeometry[] = []
  let cur = byId.get(layerId)
  while (cur) {
    if (cur.op === 'conv2d') out.push({ kind: 'conv2d', size: cur.kernel_shape![2], stride: cur.stride ?? 1 })
    else if (cur.op === 'maxpool2d') out.push({ kind: 'maxpool2d', size: cur.window!, stride: cur.stride ?? cur.window! })
    cur = cur.inputs.length ? byId.get(cur.inputs[0]) : undefined
  }
  return out.reverse()
}

export function convStages(trace: Trace): ConvStage[] {
  const byId = new Map(trace.layers.map((l) => [l.id, l]))
  return trace.layers
    .filter((l) => l.op === 'conv2d' && l.params)
    .map((conv, index) => {
      const act = consumers(trace, conv.id).find((l) => l.op === 'relu') ?? null
      const pool = consumers(trace, (act ?? conv).id).find((l) => l.op === 'maxpool2d') ?? null
      const input = byId.get(conv.inputs[0])!
      return {
        index,
        conv,
        act,
        pool,
        inShape: input.out_shape,
        inputId: input.id,
        inputRuntime: runtimeOf(trace, input.id),
        kernel: conv.params!.kernel,
        bias: conv.params!.bias,
        kernelShape: conv.kernel_shape!,
        stride: conv.stride ?? 1,
        convStack: geometryTo(trace, conv.id),
        poolStack: pool ? geometryTo(trace, pool.id) : null,
      }
    })
}

/** Ids of the layers computed after `step` steps of a layer-by-layer playback (step 0 = none). */
export function reachedLayers(trace: Trace, step: number): string[] {
  return trace.layers.slice(0, Math.max(0, step)).map((l) => l.id)
}

/** Number of learned values of one layer, from the parameter index (null when unknown). */
export function layerParamCount(trace: Trace, layer: TraceLayer): number | null {
  if (!layer.params || !trace.params) return null
  const byName = new Map(trace.params.tensors.map((t) => [t.name, t]))
  let n = 0
  for (const ref of Object.values(layer.params)) {
    const t = byName.get(ref)
    if (!t) return null
    n += prod(t.shape)
  }
  return n
}

// --------------------------------------------------------------------------- level of detail
export interface Shown {
  /** neuron indices to draw, ascending */
  indices: number[]
  total: number
  /** how many of them were picked by score, evenly spaced, or pinned by the user */
  byScore: number
  strided: number
  pinned: number
}

/**
 * Which neurons of a layer to draw when it has more than `max`: the `max - strided` best by
 * `score` (e.g. |activation|), then `strided` evenly spaced ones that were not already chosen,
 * plus every pinned index (e.g. the selected neuron). With `total <= max` every neuron is shown.
 * Nothing is invented: every returned index is a real neuron of the layer.
 */
export function selectShown(total: number, max: number, score: (i: number) => number, strided = Math.round(max / 6), pinned: number[] = []): Shown {
  if (total <= max) return { indices: Array.from({ length: total }, (_, i) => i), total, byScore: total, strided: 0, pinned: 0 }
  const want = Math.max(0, Math.min(strided, max))
  const chosen = new Set<number>()
  const order = Array.from({ length: total }, (_, i) => i).sort((a, b) => score(b) - score(a) || a - b)
  const topN = Math.max(0, max - want)
  const inTop = new Set(order.slice(0, topN))
  // pinned neurons that are not already among the best take the place of the weakest ones
  const pins = [...new Set(pinned.filter((p) => p >= 0 && p < total && !inTop.has(p)))].slice(0, max)
  for (let k = 0; k < order.length && chosen.size < Math.max(0, topN - pins.length); k++) chosen.add(order[k])
  const byScore = chosen.size
  let added = 0
  if (want > 0) {
    for (let k = 0; k < want; k++) {
      // evenly spaced positions; step past ones already taken
      let i = Math.floor(((k + 0.5) * total) / want)
      while (chosen.has(i) && i < total - 1) i++
      if (!chosen.has(i)) {
        chosen.add(i)
        added++
      }
    }
  }
  let pinnedAdded = 0
  for (const p of pins) {
    if (!chosen.has(p)) {
      chosen.add(p)
      pinnedAdded++
    }
  }
  return { indices: [...chosen].sort((a, b) => a - b), total, byScore, strided: added, pinned: pinnedAdded }
}

/** The `k` indices with the largest |value| among `candidates`, strongest first. */
export function strongest(candidates: readonly number[], value: (i: number) => number, k: number): number[] {
  return [...candidates].sort((a, b) => Math.abs(value(b)) - Math.abs(value(a)) || a - b).slice(0, k)
}

// --------------------------------------------------------------------------- sliding window
/** Output location (row, col) of the n-th kernel position in raster order. */
export const rasterPos = (n: number, width: number) => ({ r: Math.floor(n / width), q: n % width })

/**
 * Pooling output location whose window contains the input cell (r, q), or null when that cell is
 * not covered (rows/columns dropped by a floor-sized output).
 */
export function poolCell(r: number, q: number, window: number, stride: number, outH: number, outW: number): { r: number; q: number } | null {
  const pr = Math.floor(r / stride)
  const pq = Math.floor(q / stride)
  if (pr >= outH || pq >= outW) return null
  if (r - pr * stride >= window || q - pq * stride >= window) return null
  return { r: pr, q: pq }
}

/**
 * Number of pooled cells already available after the first `computed` conv outputs (raster
 * order): a pooled cell is ready when the last cell of its window has been computed.
 */
export function pooledReady(computed: number, inW: number, window: number, stride: number, outH: number, outW: number): boolean[] {
  const ready: boolean[] = []
  for (let r = 0; r < outH; r++)
    for (let q = 0; q < outW; q++) {
      const last = (r * stride + window - 1) * inW + (q * stride + window - 1)
      ready.push(last < computed)
    }
  return ready
}

/** Index of the largest value of channel `c` of a [C, H, W] tensor (row-major inside the channel). */
export function channelArgmax(t: ArrayLike<number>, c: number, h: number, w: number): number {
  const base = c * h * w
  let best = 0
  for (let i = 1; i < h * w; i++) if (t[base + i] > t[base + best]) best = i
  return best
}

/** One channel of a [C, H, W] tensor as a view. */
export const channel = (t: Float32Array, c: number, h: number, w: number) => t.subarray(c * h * w, (c + 1) * h * w)

/** Kernel [Cout, Cin, k, k]: the k x k slice for (cout, cin), or the mean |w| over the input channels. */
export function kernelSlice(K: Float32Array, shape: number[], cout: number, cin: number | 'meanabs'): Float32Array {
  const [, ci, kh, kw] = shape
  const n = kh * kw
  if (cin !== 'meanabs') return K.subarray((cout * ci + cin) * n, (cout * ci + cin + 1) * n)
  const out = new Float32Array(n)
  for (let c = 0; c < ci; c++) for (let i = 0; i < n; i++) out[i] += Math.abs(K[(cout * ci + c) * n + i]) / ci
  return out
}

/** Grid (width x height, width >= height) that holds exactly `n` cells: the divisor of n closest to sqrt(2n). */
export function gridDims(n: number): { w: number; h: number } {
  if (n <= 0) return { w: 1, h: 1 }
  const target = Math.sqrt(2 * n)
  let best = n
  for (let w = 1; w <= n; w++) if (n % w === 0 && Math.abs(w - target) < Math.abs(best - target)) best = w
  return { w: best, h: n / best }
}
