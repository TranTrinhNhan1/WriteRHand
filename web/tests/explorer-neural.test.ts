import { describe, expect, it } from 'vitest'
import type { ReplayData, Trace, TraceLayer } from '../src/engine/contract'
import { receptiveField } from '../src/engine/reconstruct'
import {
  channelArgmax,
  convStages,
  denseBlocks,
  geometryTo,
  gridDims,
  kernelSlice,
  layerParamCount,
  poolCell,
  pooledReady,
  rasterPos,
  reachedLayers,
  runtimeOf,
  selectShown,
  strongest,
} from '../src/components/explorer/neural/model'
import { column, correctCount, predicted, samplePixels, seriesAt, snapshotAt, tensorLayout, tensorMax } from '../src/components/explorer/neural/replay'
import { stepText } from '../src/components/explorer/neural/steps'
import { diff, fx } from '../src/components/explorer/neural/format'

const L = (l: Partial<TraceLayer> & Pick<TraceLayer, 'id' | 'op' | 'out_shape'>): TraceLayer => ({ inputs: [], ...l })

function trace(layers: TraceLayer[], tensors: { name: string; shape: number[] }[] = []): Trace {
  let off = 0
  return {
    schema: 'writerhand.trace/1',
    kind: 'mlp',
    preprocessing: [],
    input: { name: 'pixels', shape: ['N', 784], range: [0, 1] },
    layers,
    runtime_outputs: [],
    output: { name: 'probabilities', kind: 'softmax_probability', calibrated: true, classes: ['a', 'b', 'c'] },
    params: {
      file: 'x',
      sha256: '',
      nbytes: 0,
      tensors: tensors.map((t) => {
        const n = t.shape.reduce((a, b) => a * b, 1) * 4
        const e = { ...t, dtype: 'float32' as const, offset: off, nbytes: n, layer: '', role: '' }
        off += n
        return e
      }),
    },
    n_parameters: null,
  }
}

// an MLP with three hidden layers (the code must not assume two)
const mlp = trace(
  [
    L({ id: 'image', op: 'input', out_shape: [4, 4] }),
    L({ id: 'flatten', op: 'flatten', out_shape: [16], inputs: ['image'] }),
    L({ id: 'standardize', op: 'standardize', out_shape: [16], inputs: ['flatten'], runtime: 'x' }),
    L({ id: 'z0', op: 'dense', in_features: 16, out_features: 8, params: { weight: 'W0', bias: 'b0' }, out_shape: [8], inputs: ['standardize'], runtime: 'z0' }),
    L({ id: 'a0', op: 'relu', out_shape: [8], inputs: ['z0'], runtime: 'a0' }),
    L({ id: 'z1', op: 'dense', in_features: 8, out_features: 6, params: { weight: 'W1', bias: 'b1' }, out_shape: [6], inputs: ['a0'], runtime: 'z1' }),
    L({ id: 'a1', op: 'relu', out_shape: [6], inputs: ['z1'], runtime: 'a1' }),
    L({ id: 'z2', op: 'dense', in_features: 6, out_features: 5, params: { weight: 'W2', bias: 'b2' }, out_shape: [5], inputs: ['a1'], runtime: 'z2' }),
    L({ id: 'a2', op: 'relu', out_shape: [5], inputs: ['z2'], runtime: 'a2' }),
    L({ id: 'logits', op: 'dense', in_features: 5, out_features: 3, params: { weight: 'W3', bias: 'b3' }, out_shape: [3], inputs: ['a2'], runtime: 'logits' }),
    L({ id: 'output', op: 'softmax', out_shape: [3], inputs: ['logits'], runtime: 'probabilities' }),
  ],
  [
    { name: 'W0', shape: [16, 8] },
    { name: 'b0', shape: [8] },
    { name: 'W1', shape: [8, 6] },
    { name: 'b1', shape: [6] },
    { name: 'W2', shape: [6, 5] },
    { name: 'b2', shape: [5] },
    { name: 'W3', shape: [5, 3] },
    { name: 'b3', shape: [3] },
  ],
)

// a CNN with three conv stages, the last one without pooling, and a hidden dense layer
const cnn = trace([
  L({ id: 'image', op: 'input', out_shape: [28, 28] }),
  L({ id: 'flatten', op: 'flatten', out_shape: [784], inputs: ['image'] }),
  L({ id: 'standardize', op: 'standardize', out_shape: [784], inputs: ['flatten'], runtime: 'x' }),
  L({ id: 'reshape', op: 'reshape', out_shape: [1, 28, 28], inputs: ['standardize'] }),
  L({ id: 'conv0', op: 'conv2d', kernel_shape: [4, 1, 5, 5], stride: 1, params: { kernel: 'K0', bias: 'kb0' }, out_shape: [4, 24, 24], inputs: ['reshape'], runtime: 'conv0' }),
  L({ id: 'act0', op: 'relu', out_shape: [4, 24, 24], inputs: ['conv0'], runtime: 'act0' }),
  L({ id: 'pool0', op: 'maxpool2d', window: 2, stride: 2, out_shape: [4, 12, 12], inputs: ['act0'], runtime: 'pool0' }),
  L({ id: 'conv1', op: 'conv2d', kernel_shape: [6, 4, 3, 3], stride: 1, params: { kernel: 'K1', bias: 'kb1' }, out_shape: [6, 10, 10], inputs: ['pool0'], runtime: 'conv1' }),
  L({ id: 'act1', op: 'relu', out_shape: [6, 10, 10], inputs: ['conv1'], runtime: 'act1' }),
  L({ id: 'pool1', op: 'maxpool2d', window: 2, stride: 2, out_shape: [6, 5, 5], inputs: ['act1'], runtime: 'pool1' }),
  L({ id: 'conv2', op: 'conv2d', kernel_shape: [8, 6, 3, 3], stride: 1, params: { kernel: 'K2', bias: 'kb2' }, out_shape: [8, 3, 3], inputs: ['pool1'], runtime: 'conv2' }),
  L({ id: 'act2', op: 'relu', out_shape: [8, 3, 3], inputs: ['conv2'], runtime: 'act2' }),
  L({ id: 'flat', op: 'flatten', out_shape: [72], inputs: ['act2'] }),
  L({ id: 'h', op: 'dense', in_features: 72, out_features: 20, params: { weight: 'W0', bias: 'b0' }, out_shape: [20], inputs: ['flat'], runtime: 'h' }),
  L({ id: 'ha', op: 'relu', out_shape: [20], inputs: ['h'], runtime: 'ha' }),
  L({ id: 'logits', op: 'dense', in_features: 20, out_features: 3, params: { weight: 'W1', bias: 'b1' }, out_shape: [3], inputs: ['ha'], runtime: 'logits' }),
  L({ id: 'output', op: 'softmax', out_shape: [3], inputs: ['logits'], runtime: 'probabilities' }),
])

describe('stage grouping from trace.layers', () => {
  it('finds every dense block of an MLP of any depth, with its ReLU and runtime names', () => {
    const b = denseBlocks(mlp)
    expect(b.map((x) => x.dense.id)).toEqual(['z0', 'z1', 'z2', 'logits'])
    expect(b.map((x) => x.act?.id ?? null)).toEqual(['a0', 'a1', 'a2', null])
    expect(b.map((x) => x.inputRuntime)).toEqual(['x', 'a0', 'a1', 'a2'])
    expect(b.map((x) => x.isOutput)).toEqual([false, false, false, true])
    expect(b[1]).toMatchObject({ nIn: 8, nOut: 6, weight: 'W1', bias: 'b1', zRuntime: 'z1', aRuntime: 'a1' })
  })

  it('groups conv / ReLU / pool stages, including a stage without pooling', () => {
    const s = convStages(cnn)
    expect(s.map((x) => [x.conv.id, x.act?.id, x.pool?.id ?? null])).toEqual([
      ['conv0', 'act0', 'pool0'],
      ['conv1', 'act1', 'pool1'],
      ['conv2', 'act2', null],
    ])
    expect(s[0].inputRuntime).toBe('x') // through the reshape
    expect(s[0].inShape).toEqual([1, 28, 28])
    expect(s[2].inputRuntime).toBe('pool1')
    expect(s[1].convStack.map((g) => [g.kind, g.size, g.stride])).toEqual([
      ['conv2d', 5, 1],
      ['maxpool2d', 2, 2],
      ['conv2d', 3, 1],
    ])
    expect(s[2].poolStack).toBeNull()
    // the classifier reads the flattened conv2 activations and has a hidden layer
    const d = denseBlocks(cnn)
    expect(d.map((x) => [x.inputRuntime, x.isOutput])).toEqual([
      ['act2', false],
      ['ha', true],
    ])
  })

  it('computes receptive fields over the real stack (5x5, pool 2, 3x3 = 10x10)', () => {
    const rf = receptiveField(geometryTo(cnn, 'conv1'), 0, 0)
    expect([rf.y1 - rf.y0, rf.x1 - rf.x0]).toEqual([10, 10])
    const rf2 = receptiveField(geometryTo(cnn, 'conv1'), 2, 3)
    expect([rf2.y0, rf2.x0]).toEqual([4, 6])
  })

  it('resolves runtime outputs through shape-only layers', () => {
    expect(runtimeOf(cnn, 'reshape')).toBe('x')
    expect(runtimeOf(cnn, 'flat')).toBe('act2')
    expect(runtimeOf(cnn, 'image')).toBeNull()
  })

  it('lights layers in computation order', () => {
    expect(reachedLayers(mlp, 0)).toEqual([])
    expect(reachedLayers(mlp, 3)).toEqual(['image', 'flatten', 'standardize'])
    expect(reachedLayers(mlp, 99)).toHaveLength(mlp.layers.length)
  })

  it('counts learned values from the parameter index', () => {
    expect(layerParamCount(mlp, mlp.layers[3])).toBe(16 * 8 + 8)
    expect(layerParamCount(mlp, mlp.layers[4])).toBeNull()
  })

  it('describes each step with the layer’s own numbers', () => {
    expect(stepText(mlp.layers[5])).toEqual({ key: 'ex.shell.step.dense', params: { id: 'z1', out: '6', a: '8', b: '6' } })
    expect(stepText(cnn.layers[7]).params).toMatchObject({ cout: 6, cin: 4, k: '3×3' })
  })
})

describe('level of detail', () => {
  const act = Float32Array.from({ length: 512 }, (_, i) => (i % 7 === 0 ? i / 10 : 0))

  it('shows everything when the layer is small enough', () => {
    const s = selectShown(10, 48, (i) => i)
    expect(s.indices).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    expect(s.strided).toBe(0)
  })

  it('picks the most active neurons plus evenly spaced ones, never more than asked, all real and distinct', () => {
    const s = selectShown(512, 48, (i) => act[i], 8)
    expect(s.indices.length).toBeLessThanOrEqual(48)
    expect(new Set(s.indices).size).toBe(s.indices.length)
    expect(s.indices.every((i) => Number.isInteger(i) && i >= 0 && i < 512)).toBe(true)
    expect(s.byScore).toBe(40)
    expect(s.byScore + s.strided + s.pinned).toBe(s.indices.length)
    // the 40 strongest are all there
    const top = Array.from({ length: 512 }, (_, i) => i)
      .sort((a, b) => act[b] - act[a])
      .slice(0, 40)
    for (const i of top) expect(s.indices).toContain(i)
    expect([...s.indices].sort((a, b) => a - b)).toEqual(s.indices)
  })

  it('keeps a pinned neuron without exceeding the budget', () => {
    const s = selectShown(512, 48, (i) => act[i], 8, [1])
    expect(s.indices).toContain(1)
    expect(s.indices.length).toBeLessThanOrEqual(48)
    expect(s.pinned).toBe(1)
    // a pinned neuron that is already among the best costs nothing
    const best = Array.from({ length: 512 }, (_, i) => i).sort((a, b) => act[b] - act[a])[0]
    expect(selectShown(512, 48, (i) => act[i], 8, [best]).pinned).toBe(0)
  })

  it('ranks the strongest contributions by magnitude', () => {
    const v = [0.1, -3, 2, -0.5]
    expect(strongest([0, 1, 2, 3], (i) => v[i], 2)).toEqual([1, 2])
  })

  it('lays out a whole layer as an exact grid', () => {
    expect(gridDims(512)).toEqual({ w: 32, h: 16 })
    expect(gridDims(256)).toEqual({ w: 16, h: 16 })
    expect(gridDims(10)).toEqual({ w: 5, h: 2 })
    for (const n of [1, 7, 100, 400, 784]) {
      const g = gridDims(n)
      expect(g.w * g.h).toBe(n)
    }
  })
})

describe('sliding window', () => {
  it('walks output locations in raster order', () => {
    expect([0, 1, 25, 26, 675].map((n) => rasterPos(n, 26))).toEqual([
      { r: 0, q: 0 },
      { r: 0, q: 1 },
      { r: 0, q: 25 },
      { r: 1, q: 0 },
      { r: 25, q: 25 },
    ])
  })

  it('maps a conv cell to its pooling window, or to none when the pool drops it', () => {
    expect(poolCell(5, 7, 2, 2, 13, 13)).toEqual({ r: 2, q: 3 })
    expect(poolCell(10, 3, 2, 2, 5, 5)).toBeNull() // 11 x 11 -> 5 x 5 drops row 10
  })

  it('reveals a pooled cell only once its whole window has been computed', () => {
    const ready = pooledReady(4 * 4, 4, 2, 2, 2, 2) // 4 x 4 map, first full row pair done after 8 cells
    expect(ready).toEqual([true, true, true, true])
    expect(pooledReady(5, 4, 2, 2, 2, 2)).toEqual([false, false, false, false])
    expect(pooledReady(6, 4, 2, 2, 2, 2)).toEqual([true, false, false, false])
  })

  it('finds the strongest location of a channel and slices kernels', () => {
    const t = Float32Array.from([0, 1, 2, 3, 9, 1, 0, 0])
    expect(channelArgmax(t, 1, 2, 2)).toBe(0)
    const K = Float32Array.from({ length: 2 * 2 * 4 }, (_, i) => (i % 2 ? -i : i))
    expect(Array.from(kernelSlice(K, [2, 2, 2, 2], 1, 0))).toEqual([8, -9, 10, -11])
    expect(Array.from(kernelSlice(K, [2, 2, 2, 2], 1, 'meanabs'))).toEqual([10, 11, 12, 13])
  })
})

describe('training replay helpers (in-test fixture only)', () => {
  const r: ReplayData = {
    schema: 'writerhand.replay/1',
    dataset: 'd',
    model: 'mlp',
    seed: 1,
    params: {},
    epochs: [1, 2, 3],
    series: { loss: [0.9, 0.5, 0.4], val_acc: [0.6, 0.7, 0.72] },
    best_epoch: 2,
    stopped_epoch: 3,
    samples: { indices: [5, 9], labels: [0, 2], pixels: [Array(4).fill(255), Array(4).fill(0)] },
    snapshots: [
      { epoch: 0, metrics: {}, proba: [[0.3, 0.3, 0.4], [0, 0, 0]], tensors: { W: { shape: [4, 2], data: [1, -2, 3, 0, 0, 0, 0, 0], note: 'W[:, :2]' } } },
      { epoch: 2, metrics: { loss: 0.5 }, proba: [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], tensors: { W: { shape: [4, 2], data: [4, 0, 0, 0, 0, 0, 0, -5] } } },
    ],
    deployed_epoch: 2,
    final_matches_deployed: true,
    note: '',
  }

  it('reads series at an epoch and reports the initial snapshot as unmeasured', () => {
    expect(seriesAt(r, 2)).toEqual({ loss: 0.5, val_acc: 0.7 })
    expect(seriesAt(r, 0)).toBeNull()
    expect(snapshotAt(r, r.deployed_epoch)).toBe(1)
    expect(snapshotAt(r, null)).toBe(-1)
  })

  it('chooses a layout from the saved shape', () => {
    expect(tensorLayout({ shape: [784, 16], data: [] }, 28)).toEqual({ kind: 'columns', side: 28, rows: 784, cols: 16 })
    expect(tensorLayout({ shape: [8, 1, 3, 3], data: [] }, 28)).toMatchObject({ kind: 'kernels', cout: 8, cin: 1 })
    expect(tensorLayout({ shape: [10, 20], data: [] }, 28)).toEqual({ kind: 'matrix', w: 20, h: 10 })
    expect(tensorLayout({ shape: [10], data: [] }, 28)).toEqual({ kind: 'vector', n: 10 })
    expect(Array.from(column([1, 2, 3, 4, 5, 6], 3, 2, 1))).toEqual([2, 4, 6])
  })

  it('uses one colour scale across snapshots and counts correct predictions', () => {
    expect(tensorMax(r.snapshots, 'W')).toBe(5)
    expect(correctCount(r.snapshots[0], r.samples.labels)).toBe(0)
    expect(correctCount(r.snapshots[1], r.samples.labels)).toBe(2)
    expect(predicted([0, 0, 0])).toBe(-1) // no votes yet: no prediction
    expect(Array.from(samplePixels([0, 255, 51]))).toEqual([0, 1, Math.fround(0.2)])
  })
})

describe('number formatting', () => {
  it('uses a real minus sign and short exponents', () => {
    expect(fx(-0.5, 2)).toBe('−0.50')
    expect(fx(0.5, 2, true)).toBe('+0.50')
    expect(diff(0)).toBe('0')
    expect(diff(2.5e-7)).toBe('2.5e−7')
  })
})
