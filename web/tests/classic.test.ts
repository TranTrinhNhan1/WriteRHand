import { describe, expect, it } from 'vitest'
import type { AdaBoostEnsemble, Tree } from '../src/engine/adaboost'
import { modelsOf } from '../src/engine/contract'
import { nbClassTerms, softmaxClassTerms } from '../src/engine/reconstruct'
import {
  absQuantile,
  classRow,
  compactTree,
  curveDomain,
  fmt,
  fmtDiff,
  gaussianPdf,
  leaderChanges,
  makeAdditive,
  pixelOrder,
  routeUsage,
  rowLeader,
  runningScores,
  sampleCurve,
  shareSeries,
  splitUsagePrefix,
  treeSplitCounts,
} from '../src/components/explorer/classic/model'
import { hasBundle, loadManifest, makeRegistry, readJson, toPixels01, type TestVectors } from './bundle'

// A small fixed additive model: 3 classes x 4 pixels (test input only, never drawn).
const T = [
  [1, -2, 0.5, 0],
  [-1, 3, 0, 0.25],
  [0, 0, -4, 2],
]
const START = [0.1, -0.2, 0.3]
const additive = () => makeAdditive(3, 4, (k) => ({ terms: T[k], start: START[k] }))

describe('additive accumulation', () => {
  it('totals are start + sum of the terms', () => {
    const a = additive()
    for (let k = 0; k < 3; k++) expect(a.total[k]).toBeCloseTo(START[k] + T[k].reduce((s, v) => s + v, 0), 12)
    expect(Array.from(classRow(a, 1))).toEqual(T[1])
  })

  it('magnitude order sorts by decreasing |term| of the chosen class; raster keeps pixel order', () => {
    const a = additive()
    const order = Array.from(pixelOrder('magnitude', classRow(a, 0)))
    expect(order).toEqual([1, 0, 2, 3])
    expect(Array.from(pixelOrder('raster', classRow(a, 0)))).toEqual([0, 1, 2, 3])
    expect([...order].sort()).toEqual([0, 1, 2, 3])
  })

  it('running sums start at the start values, follow the order and end at the totals for any order', () => {
    const a = additive()
    for (const kind of ['magnitude', 'raster'] as const) {
      for (let k = 0; k < 3; k++) {
        const order = pixelOrder(kind, classRow(a, k))
        const rows = runningScores(a, order)
        expect(rows.length).toBe((order.length + 1) * 3)
        for (let c = 0; c < 3; c++) {
          expect(rows[c]).toBe(START[c])
          let s = START[c]
          for (let i = 0; i < order.length; i++) {
            s += T[c][order[i]]
            expect(rows[(i + 1) * 3 + c]).toBeCloseTo(s, 12)
          }
          expect(rows[order.length * 3 + c]).toBeCloseTo(a.total[c], 12)
        }
      }
    }
  })

  it('finds the leader at every step and the steps where it changes', () => {
    const a = additive()
    const rows = runningScores(a, pixelOrder('raster', classRow(a, 0)))
    // step 0: 0.1 -0.2 0.3 -> class 2; step 1: 1.1 -1.2 0.3 -> 0; step 2: -0.9 1.8 0.3 -> 1; step 3: -0.4 1.8 -3.7 -> 1; step 4: -0.4 2.05 -1.7 -> 1
    expect([0, 1, 2, 3, 4].map((s) => rowLeader(rows, 3, s))).toEqual([2, 0, 1, 1, 1])
    expect(leaderChanges(rows, 3)).toEqual([
      { step: 0, klass: 2 },
      { step: 1, klass: 0 },
      { step: 2, klass: 1 },
    ])
    // an all-zero row has no leader when asked so (AdaBoost before the first vote)
    expect(rowLeader(new Float64Array([0, 0, 0]), 3, 0, true)).toBe(-1)
  })

  it('absQuantile picks a value of the data', () => {
    expect(absQuantile([1, -5, 2, 3], 1)).toBe(5)
    expect(absQuantile([1, -5, 2, 3], 0)).toBe(1)
  })
})

describe('Gaussian curve helpers', () => {
  it('peak is 1/sqrt(2 pi var) at the mean, and the domain covers the input and +-4 sd', () => {
    expect(gaussianPdf(0.3, 0.3, 0.25)).toBeCloseTo(1 / Math.sqrt(2 * Math.PI * 0.25), 12)
    const [lo, hi] = curveDomain([0, 1], [1, 0.25], 9)
    expect(lo).toBeLessThanOrEqual(-4)
    expect(hi).toBeGreaterThanOrEqual(9)
    const ys = sampleCurve(0, 1, -1, 1, 3)
    expect(ys[1]).toBeCloseTo(gaussianPdf(0, 0, 1), 12)
    expect(ys[0]).toBeCloseTo(ys[2], 12)
  })
})

// A fixed 9-node tree (test input only):            0
//                                               1       2
//                                             3   4   5   6
//                                                    7 8
function smallTree(): Tree {
  const f = Int16Array.from([10, 11, 12, -1, -1, 13, -1, -1, -1])
  const l = Int16Array.from([1, 3, 5, -1, -1, 7, -1, -1, -1])
  const r = Int16Array.from([2, 4, 6, -1, -1, 8, -1, -1, -1])
  const c = Uint8Array.from([0, 0, 0, 1, 2, 0, 3, 4, 5])
  const t = Float64Array.from([0.5, 0.1, 0.2, 0, 0, 0.3, 0, 0, 0])
  return { f, l, r, c, t }
}
function sizesOf(tree: Tree): Int32Array {
  const s = new Int32Array(tree.f.length)
  for (let i = tree.f.length - 1; i >= 0; i--) s[i] = tree.f[i] < 0 ? 1 : 1 + s[tree.l[i]] + s[tree.r[i]]
  return s
}

describe('tree compaction', () => {
  it('draws the top levels and the route, folds the rest, and the counts add up', () => {
    const tree = smallTree()
    const sizes = sizesOf(tree)
    const route = [0, 2, 5, 8]
    for (const depth of [0, 1, 2, 3]) {
      const ct = compactTree(tree, sizes, route, depth, (v) => v)
      expect(ct.shown + ct.collapsed).toBe(ct.total)
      expect(ct.total).toBe(9)
      const drawn = ct.nodes.filter((n) => n.kind !== 'stub')
      const stubs = ct.nodes.filter((n) => n.kind === 'stub')
      expect(drawn.length).toBe(ct.shown)
      expect(stubs.reduce((s, n) => s + n.count, 0)).toBe(ct.collapsed)
      for (const id of route) expect(drawn.some((n) => n.node === id && n.onRoute)).toBe(true)
      // every stub stands for exactly its real subtree
      for (const s of stubs) expect(s.count).toBe(sizes[s.node])
      // parents are drawn split nodes listed before their children
      for (const [i, n] of ct.nodes.entries()) if (n.parent >= 0) expect(ct.nodes[n.parent].kind === 'split' && n.parent < i).toBe(true)
    }
    const fold = compactTree(tree, sizes, route, 0, (v) => v)
    // root + route (2, 5, 8) drawn; node 1 (3 nodes), 6, 7 folded
    expect(fold.shown).toBe(4)
    expect(fold.nodes.filter((n) => n.kind === 'stub').map((n) => n.count)).toEqual([3, 1, 1])
    const full = compactTree(tree, sizes, route, 9, (v) => v)
    expect(full.collapsed).toBe(0)
    // leaves and stubs get consecutive slots left to right
    const slots = full.nodes.filter((n) => n.kind !== 'split').map((n) => n.x)
    expect(slots).toEqual([0, 1, 2, 3, 4])
  })
})

describe('pixel usage and vote shares', () => {
  it('split counts, prefix counts and shares are consistent', () => {
    const t1 = smallTree()
    const ens = { trees: [t1, t1], alphas: [0.5, 0.25], classes: [0, 1, 2, 3, 4, 5] } as unknown as AdaBoostEnsemble
    const counts = treeSplitCounts(t1, 16)
    expect(counts[10] + counts[11] + counts[12] + counts[13]).toBe(4)
    const prefix = splitUsagePrefix(ens, 16)
    expect(prefix.length).toBe(3 * 16)
    expect(prefix.subarray(0, 16).every((v) => v === 0)).toBe(true)
    expect(prefix[2 * 16 + 13]).toBe(2)
    const cumulative = [new Float64Array([0, 0]), new Float64Array([0.5, 0]), new Float64Array([0.5, 0.25])]
    const sh = shareSeries(cumulative, 2)
    expect(Array.from(sh)).toEqual([0, 0, 1, 0, 0.5 / 0.75, 0.25 / 0.75])
  })
})

describe('formatting', () => {
  it('uses a real minus sign and short scientific notation for tiny differences', () => {
    expect(fmt(-1.23456, 2)).toBe('−1.23')
    expect(fmt(-0.0001, 2)).toBe('0.00')
    expect(fmtDiff(0)).toBe('0')
    expect(fmtDiff(3.14e-6)).toBe('3.1e−6')
  })
})

describe.skipIf(!hasBundle)('classic views rebuild the deployed outputs', () => {
  it('Naive Bayes and softmax totals match the graph logits; AdaBoost compaction and usage add up on the real ensemble', async () => {
    const manifest = loadManifest()
    const registry = makeRegistry(manifest)
    for (const [dsId, ds] of Object.entries(manifest.datasets)) {
      const vectors = readJson<TestVectors>(ds.test_vectors)
      const px = toPixels01(vectors.pixels[2])
      const models = modelsOf(ds)
      if (models.includes('naive_bayes')) {
        const lm = await registry.load(dsId, 'naive_bayes')
        const ex = await registry.execute(dsId, 'naive_bayes', px)
        const layer = lm.trace.layers.find((l) => l.op === 'gaussian_loglik')!
        const p = lm.params!
        const [mean, variance, prior] = [p.get(layer.params!.mean), p.get(layer.params!.variance), p.get(layer.params!.log_prior)]
        const K = ds.classes.length
        const N = layer.n_features!
        const a = makeAdditive(K, N, (k) => {
          const r = nbClassTerms(ex.outputs.x, mean, variance, prior, k, N)
          return { terms: r.perPixel, start: r.logPrior }
        })
        const rows = runningScores(a, pixelOrder('magnitude', classRow(a, ex.top)))
        for (let k = 0; k < K; k++) {
          expect(Math.abs(a.total[k] - ex.outputs.logits[k])).toBeLessThan(0.05)
          expect(rows[N * K + k]).toBeCloseTo(a.total[k], 6)
        }
      }
      if (models.includes('softmax')) {
        const lm = await registry.load(dsId, 'softmax')
        const ex = await registry.execute(dsId, 'softmax', px)
        const layer = lm.trace.layers.find((l) => l.op === 'dense')!
        const [W, b] = [lm.params!.get(layer.params!.weight), lm.params!.get(layer.params!.bias)]
        const a = makeAdditive(layer.out_features!, layer.in_features!, (k) => {
          const s = softmaxClassTerms(ex.outputs.x, W, b, layer.in_features!, layer.out_features!, k)
          return { terms: s.terms, start: s.bias }
        })
        for (let k = 0; k < a.K; k++) expect(Math.abs(a.total[k] - ex.outputs.logits[k])).toBeLessThan(5e-3)
      }
      if (models.includes('adaboost')) {
        const lm = await registry.load(dsId, 'adaboost')
        const ens = lm.ensemble!
        const ex = await registry.execute(dsId, 'adaboost', px)
        const run = ex.ada!
        const nPix = px.length
        for (const t of [0, Math.floor(ens.trees.length / 2), ens.trees.length - 1]) {
          const tree = ens.trees[t]
          const sizes = sizesOf(tree)
          const r = run.rounds[t]
          for (const depth of [1, 2]) {
            const ct = compactTree(tree, sizes, [...r.path.map((s) => s.node), r.leaf], depth, (v) => v)
            expect(ct.shown + ct.collapsed).toBe(ens.nodeCounts[t])
            expect(ct.nodes.find((n) => n.kind === 'leaf' && n.onRoute)!.klass).toBe(r.klass)
          }
        }
        const prefix = splitUsagePrefix(ens, nPix)
        let splits = 0
        for (const tree of ens.trees) for (let i = 0; i < tree.f.length; i++) if (tree.f[i] >= 0) splits++
        expect(prefix.subarray(ens.trees.length * nPix).reduce((s, v) => s + v, 0)).toBe(splits)
        const use = routeUsage(run, run.rounds.length, nPix)
        expect(use.reduce((s, v) => s + v, 0)).toBe(run.rounds.reduce((s, r) => s + r.path.length, 0))
        const sh = shareSeries(run.cumulative, ds.classes.length)
        const last = sh.subarray(run.rounds.length * ds.classes.length)
        for (let k = 0; k < last.length; k++) expect(last[k]).toBeCloseTo(run.shares[k], 12)
      }
    }
  })
})
