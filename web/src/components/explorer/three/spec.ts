/**
 * Data side of the 3D network views (no three.js here, so it is unit-tested in Node).
 * Every layer, width, channel count and map size is read from the validated trace and the runtime outputs
 * of the current execution; nothing about the architecture is hard-coded.
 */
import type { Trace } from '../../../engine/contract'
import { convStages, denseBlocks, runtimeOf } from '../neural/model'

/** The learned tensors, by name (ParamSet). */
export interface ParamStore {
  get(name: string): Float32Array
}

/** Most nodes drawn for one layer; a larger layer shows its strongest ones and a "+N" marker. */
export const MAX_NODES = 1024
/** Strongest connections drawn between two layers (of all |a_i · W_ij| for the current input). */
export const MAX_EDGES = 260

export interface NodeLayer3D {
  kind: 'nodes'
  key: string
  /** short label, e.g. "h1 · 512" */
  label: string
  /** every value of the layer for the current input */
  values: ArrayLike<number>
  /** indices drawn (all of them unless the layer has more than MAX_NODES) */
  shown: number[]
  total: number
  /** signed colour scale (standardised input, logits) or >= 0 (ReLU) */
  signed: boolean
  /** class names written next to the nodes (output layer) */
  names?: string[]
  /** trace layers this column stands for: it is revealed once the playback reached the first one */
  layerIds: string[]
  /** column index of the 2D view / inspector this layer maps to (selection) */
  col: number
}

export interface MapLayer3D {
  kind: 'maps'
  key: string
  label: string
  /** channels × height × width of the drawn tensor */
  c: number
  h: number
  w: number
  data: ArrayLike<number>
  scale: 'signed' | 'positive' | 'gray'
  layerIds: string[]
  /** conv stage this stack belongs to (CNN selection), or the column for an MLP input */
  stage: number | null
  col: number | null
}

export type Layer3D = NodeLayer3D | MapLayer3D

export interface Edge3D {
  /** source index within the source layer (neuron, or pixel of a single-channel map) */
  i: number
  /** target neuron */
  j: number
  /** a_i · W_ij for the current input */
  v: number
}

export interface Edges3D {
  from: number
  to: number
  list: Edge3D[]
  /** connections that exist between the two layers (nIn × nOut) */
  total: number
  /** dense layer whose weighted sums these are (revealed with it) */
  layerId: string
}

export interface NetSpec {
  layers: Layer3D[]
  edges: Edges3D[]
}

/** Indices of the `k` largest |score(i)| over `0..n-1` (ties by index), ascending. */
export function topIndices(n: number, k: number, score: (i: number) => number): number[] {
  if (n <= k) return Array.from({ length: n }, (_, i) => i)
  const idx = Array.from({ length: n }, (_, i) => i)
  idx.sort((a, b) => Math.abs(score(b)) - Math.abs(score(a)) || a - b)
  return idx.slice(0, k).sort((a, b) => a - b)
}

/**
 * The `k` connections with the largest |src_i · W_ij| between the drawn source and target nodes.
 * `W` is row-major [nIn, nOut] as in the parameter file. A bounded min-heap keeps it O(n log k).
 */
export function strongestEdges(src: ArrayLike<number>, W: ArrayLike<number>, nOut: number, from: readonly number[], to: readonly number[], k: number): Edge3D[] {
  const heap: Edge3D[] = []
  const mag = (e: Edge3D) => Math.abs(e.v)
  const less = (a: Edge3D, b: Edge3D) => mag(a) < mag(b) || (mag(a) === mag(b) && (a.i > b.i || (a.i === b.i && a.j > b.j)))
  const up = (n: number) => {
    while (n > 0) {
      const p = (n - 1) >> 1
      if (!less(heap[n], heap[p])) break
      ;[heap[n], heap[p]] = [heap[p], heap[n]]
      n = p
    }
  }
  const down = (n: number) => {
    for (;;) {
      const l = 2 * n + 1
      const r = l + 1
      let m = n
      if (l < heap.length && less(heap[l], heap[m])) m = l
      if (r < heap.length && less(heap[r], heap[m])) m = r
      if (m === n) return
      ;[heap[n], heap[m]] = [heap[m], heap[n]]
      n = m
    }
  }
  if (k <= 0) return []
  for (const i of from) {
    const a = src[i]
    if (a === 0) continue
    const row = i * nOut
    for (const j of to) {
      const v = a * W[row + j]
      if (v === 0) continue
      const e = { i, j, v }
      if (heap.length < k) {
        heap.push(e)
        up(heap.length - 1)
      } else if (less(heap[0], e)) {
        heap[0] = e
        down(0)
      }
    }
  }
  return heap.sort((a, b) => mag(b) - mag(a) || a.i - b.i || a.j - b.j)
}

function nodeLayer(key: string, label: string, values: ArrayLike<number>, signed: boolean, layerIds: string[], col: number, names?: string[]): NodeLayer3D {
  const total = values.length
  return { kind: 'nodes', key, label, values, total, shown: topIndices(total, MAX_NODES, (i) => values[i]), signed, layerIds, col, names }
}

/** Short layer name used by the MLP views: x, h1, h2, ..., z. */
export const mlpName = (c: number, isOutput: boolean) => (c === 0 ? 'x' : isOutput ? 'z' : `h${c}`)

/**
 * MLP: the standardised input (as the image it came from when it has side² values), then one node layer per
 * dense block (its activation, or its logits for the output), with the strongest a·w connections between.
 */
export function mlpSpec(trace: Trace, params: ParamStore, outputs: Record<string, Float32Array>, classes: string[]): NetSpec {
  const blocks = denseBlocks(trace)
  const image = trace.layers.find((l) => l.op === 'input')
  const std = trace.layers.find((l) => l.op === 'standardize')
  const flat = trace.layers.find((l) => l.op === 'flatten')
  const x = outputs[blocks[0].inputRuntime ?? 'x']
  const side = image && image.out_shape.length === 2 ? image.out_shape : null
  const firstIds = [image?.id, flat?.id, std?.id, blocks[0].inputId].filter((s): s is string => !!s)
  const layers: Layer3D[] = []
  if (side && side[0] * side[1] === x.length) {
    layers.push({ kind: 'maps', key: 'x', label: `x · ${x.length}`, c: 1, h: side[0], w: side[1], data: x, scale: 'signed', layerIds: firstIds.slice(-1), stage: null, col: 0 })
  } else {
    layers.push(nodeLayer('x', `x · ${x.length}`, x, true, firstIds.slice(-1), 0))
  }
  blocks.forEach((b, k) => {
    const v = outputs[b.aRuntime ?? b.zRuntime ?? '']
    const name = mlpName(k + 1, b.isOutput)
    layers.push(nodeLayer(name, `${name} · ${v.length}`, v, !b.act, [b.act?.id ?? b.dense.id], k + 1, b.isOutput ? classes : undefined))
  })
  const edges: Edges3D[] = blocks.map((b, k) => {
    const src = layers[k]
    const dst = layers[k + 1] as NodeLayer3D
    const from = src.kind === 'nodes' ? src.shown : Array.from({ length: b.nIn }, (_, i) => i)
    return { from: k, to: k + 1, list: strongestEdges(outputs[b.inputRuntime ?? 'x'], params.get(b.weight), b.nOut, from, dst.shown, MAX_EDGES), total: b.nIn * b.nOut, layerId: b.dense.id }
  })
  return { layers, edges }
}

/**
 * CNN: the input image, one stack of feature maps per conv stage (after its ReLU when there is one), the
 * pooled stack, then the flattened vector and every dense layer as nodes, with the strongest a·w connections.
 */
export function cnnSpec(trace: Trace, params: ParamStore, outputs: Record<string, Float32Array>, pixels: Float32Array, classes: string[]): NetSpec {
  const stages = convStages(trace)
  const blocks = denseBlocks(trace)
  const image = trace.layers.find((l) => l.op === 'input')
  const layers: Layer3D[] = []
  if (image) {
    const [h, w] = image.out_shape.length === 2 ? image.out_shape : [image.out_shape[image.out_shape.length - 2], image.out_shape[image.out_shape.length - 1]]
    layers.push({ kind: 'maps', key: 'input', label: `${image.id} · ${h}×${w}`, c: 1, h, w, data: pixels, scale: 'gray', layerIds: [image.id], stage: null, col: null })
  }
  stages.forEach((st, s) => {
    const [c, h, w] = st.conv.out_shape
    const top = st.act ?? st.conv
    const rt = top.runtime ?? st.conv.runtime ?? runtimeOf(trace, top.id) ?? ''
    layers.push({
      kind: 'maps',
      key: top.id,
      label: `${st.act ? `${st.conv.id}+${st.act.id}` : st.conv.id} · ${c}×${h}×${w}`,
      c,
      h,
      w,
      data: outputs[rt],
      scale: st.act ? 'positive' : 'signed',
      layerIds: [st.conv.id],
      stage: s,
      col: null,
    })
    if (st.pool && st.pool.runtime) {
      const [pc, ph, pw] = st.pool.out_shape
      layers.push({ kind: 'maps', key: st.pool.id, label: `${st.pool.id} · ${pc}×${ph}×${pw}`, c: pc, h: ph, w: pw, data: outputs[st.pool.runtime], scale: 'positive', layerIds: [st.pool.id], stage: s, col: null })
    }
  })
  const edges: Edges3D[] = []
  blocks.forEach((b, k) => {
    if (k === 0) {
      const input = outputs[b.inputRuntime ?? '']
      layers.push(nodeLayer(b.inputId, `${b.inputId} · ${input.length}`, input, false, [b.inputId], -1))
    }
    const v = outputs[b.aRuntime ?? b.zRuntime ?? '']
    layers.push(nodeLayer(b.dense.id, `${b.dense.id} · ${v.length}`, v, !b.act, [b.act?.id ?? b.dense.id], k, b.isOutput ? classes : undefined))
    const src = layers[layers.length - 2] as NodeLayer3D
    const dst = layers[layers.length - 1] as NodeLayer3D
    edges.push({ from: layers.length - 2, to: layers.length - 1, list: strongestEdges(outputs[b.inputRuntime ?? ''], params.get(b.weight), b.nOut, src.shown, dst.shown, MAX_EDGES), total: b.nIn * b.nOut, layerId: b.dense.id })
  })
  return { layers, edges }
}

/** Grid of `n` cells, about twice as wide (depth) as tall, so a layer stays low; a column for at most 16 named outputs. */
export function nodeGrid(n: number, named: boolean): { cols: number; rows: number } {
  if (named && n <= 16) return { cols: 1, rows: n }
  const cols = Math.max(1, Math.ceil(Math.sqrt(n * 2)))
  return { cols, rows: Math.ceil(n / cols) }
}
