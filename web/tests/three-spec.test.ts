import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { Trace } from '../src/engine/contract'
import { MAX_EDGES, MAX_NODES, cnnSpec, mlpSpec, nodeGrid, strongestEdges, topIndices } from '../src/components/explorer/three/spec'

const manifest = JSON.parse(readFileSync(resolve(__dirname, '../public/data/manifest.json'), 'utf8'))

/** deterministic pseudo-random values in [-1, 1) */
function rand(n: number, seed: number): Float32Array {
  const out = new Float32Array(n)
  let s = seed >>> 0
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    out[i] = s / 2 ** 31 - 1
  }
  return out
}

/** runtime outputs and parameter tensors of the right sizes, read from the trace only */
function fake(trace: Trace) {
  const outputs: Record<string, Float32Array> = {}
  trace.layers.forEach((l, k) => {
    if (l.runtime) outputs[l.runtime] = rand(l.out_shape.reduce((a, b) => a * b, 1), k + 1).map((v) => (l.op === 'relu' || l.op === 'maxpool2d' ? Math.max(0, v) : v))
  })
  const tensors = new Map(trace.params!.tensors.map((t, k) => [t.name, rand(t.shape.reduce((a, b) => a * b, 1), 100 + k)]))
  return { outputs, params: { get: (n: string) => tensors.get(n)! } }
}

describe('3D network data', () => {
  it('top-k connections equal a brute-force ranking', () => {
    const src = rand(7, 1)
    const W = rand(7 * 5, 2)
    const all = []
    for (let i = 0; i < 7; i++) for (let j = 0; j < 5; j++) all.push({ i, j, v: src[i] * W[i * 5 + j] })
    all.sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    const got = strongestEdges(src, W, 5, [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4], 9)
    expect(got.map((e) => [e.i, e.j])).toEqual(all.slice(0, 9).map((e) => [e.i, e.j]))
    expect(topIndices(5, 10, (i) => i)).toEqual([0, 1, 2, 3, 4])
    expect(topIndices(5, 2, (i) => [0, -9, 1, 8, 2][i])).toEqual([1, 3])
  })

  for (const [ds, entry] of Object.entries<{ models: Record<string, { trace: Trace }>; classes: string[] }>(manifest.datasets)) {
    it(`MLP of ${ds}: one layer per trace block, widths from the trace`, () => {
      const trace = entry.models.mlp.trace
      const { outputs, params } = fake(trace)
      const spec = mlpSpec(trace, params, outputs, entry.classes)
      const dense = trace.layers.filter((l) => l.op === 'dense')
      expect(spec.layers).toHaveLength(dense.length + 1)
      expect(spec.edges).toHaveLength(dense.length)
      dense.forEach((d, k) => {
        const layer = spec.layers[k + 1]
        expect(layer.kind).toBe('nodes')
        if (layer.kind === 'nodes') {
          expect(layer.total).toBe(d.out_features)
          expect(layer.shown.length).toBe(Math.min(MAX_NODES, d.out_features!))
        }
        expect(spec.edges[k].total).toBe(d.in_features! * d.out_features!)
        expect(spec.edges[k].list.length).toBeLessThanOrEqual(MAX_EDGES)
      })
      const x = spec.layers[0]
      expect(x.kind === 'maps' ? x.h * x.w : x.total).toBe(dense[0].in_features)
    })

    it(`CNN of ${ds}: one stack per conv / pool layer with channels × H × W of the trace`, () => {
      const trace = entry.models.cnn.trace
      const { outputs, params } = fake(trace)
      const pixels = rand(784, 9).map((v) => Math.abs(v))
      const spec = cnnSpec(trace, params, outputs, pixels, entry.classes)
      const maps = spec.layers.filter((l) => l.kind === 'maps').slice(1)
      const want = trace.layers.filter((l) => l.op === 'conv2d' || l.op === 'maxpool2d')
      expect(maps.map((m) => (m.kind === 'maps' ? [m.c, m.h, m.w] : []))).toEqual(want.map((l) => l.out_shape))
      const dense = trace.layers.filter((l) => l.op === 'dense')
      expect(spec.edges).toHaveLength(dense.length)
      const last = spec.layers[spec.layers.length - 1]
      expect(last.kind === 'nodes' && last.names).toEqual(entry.classes)
    })
  }

  it('named outputs are a single column, other layers about twice as wide as tall', () => {
    expect(nodeGrid(10, true)).toEqual({ cols: 1, rows: 10 })
    const g = nodeGrid(512, false)
    expect(g.cols * g.rows).toBeGreaterThanOrEqual(512)
  })
})
