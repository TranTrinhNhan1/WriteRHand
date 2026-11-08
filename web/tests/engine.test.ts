import { describe, expect, it } from 'vitest'
import { parseBundleManifest, ContractError, modelsOf, type ModelKind } from '../src/engine/contract'
import { matches } from '../src/engine/store'
import { subtreeSizes } from '../src/engine/adaboost'
import * as R from '../src/engine/reconstruct'
import { bundleRequired, fileFetch, hasBundle, loadManifest, makeRegistry, readJson, toPixels01, type TestVectors } from './bundle'

const TOL_PROBA: Record<ModelKind, number> = { naive_bayes: 1e-3, softmax: 1e-4, mlp: 1e-4, cnn: 1e-4, adaboost: 5.01e-7 }

// These tests need the model bundle; CI fetches the pinned Hugging Face revision first and fails
// (instead of skipping) when it is missing, see scripts/check-bundle.mjs and the `web` job.
describe.runIf(bundleRequired)('model bundle is present', () => {
  it('exists', () => expect(hasBundle).toBe(true))
})

describe.skipIf(!hasBundle)('manifest contract', () => {
  const raw = () => structuredClone(readJson<Record<string, unknown>>('manifest.json'))

  it('accepts the exported manifest and lists models from the file, not from a constant', () => {
    const m = loadManifest()
    expect(Object.keys(m.datasets).length).toBeGreaterThan(0)
    for (const ds of Object.values(m.datasets)) expect(modelsOf(ds).length).toBeGreaterThan(0)
  })

  it('rejects a wrong schema, an inconsistent shape and a tampered parameter index', () => {
    const bad = raw() as never as { schema: string }
    bad.schema = 'other/1'
    expect(() => parseBundleManifest(bad)).toThrow(ContractError)

    const m = raw() as never as { datasets: Record<string, { models: Record<string, { trace: { layers: { op: string; out_features?: number; out_shape: number[] }[]; params: { tensors: { nbytes: number }[] } } }> }> }
    const ds = Object.values(m.datasets)[0]
    const mlp = ds.models.mlp?.trace
    if (mlp) {
      const copy = structuredClone(m)
      const dense = Object.values(copy.datasets)[0].models.mlp.trace.layers.find((l) => l.op === 'dense')!
      dense.out_features! += 1
      expect(() => parseBundleManifest(copy)).toThrow(ContractError)
      const copy2 = structuredClone(m)
      Object.values(copy2.datasets)[0].models.mlp.trace.params.tensors[0].nbytes += 4
      expect(() => parseBundleManifest(copy2)).toThrow(ContractError)
    }
  })
})

describe.skipIf(!hasBundle)('browser inference and trace reconstruction match Python', () => {
  const manifest = hasBundle ? loadManifest() : null
  const registry = hasBundle ? makeRegistry(manifest!) : null

  for (const [dsId, ds] of Object.entries(manifest?.datasets ?? {})) {
    const vectors = readJson<TestVectors>(ds.test_vectors)
    for (const model of modelsOf(ds)) {
      it(`${dsId}/${model}: predictions, logits and trace values`, async () => {
        const lm = await registry!.load(dsId, model)
        const exp = vectors.expected[model]
        const n = Math.min(vectors.pixels.length, 12)
        let worst = 0
        let reconWorst = 0
        for (let i = 0; i < n; i++) {
          const px = toPixels01(vectors.pixels[i])
          const ex = await registry!.execute(dsId, model, px)
          expect(ex.top).toBe(exp.pred[i])
          for (let k = 0; k < ex.probs.length; k++) worst = Math.max(worst, Math.abs(ex.probs[k] - exp.proba[i][k]))
          if (exp.logits) {
            const lg = ex.outputs.logits
            for (let k = 0; k < lg.length; k++) expect(Math.abs(lg[k] - exp.logits[i][k])).toBeLessThan(model === 'naive_bayes' ? 0.05 : 5e-3)
          }
          reconWorst = Math.max(reconWorst, reconstructionError(model, lm, ex))
        }
        expect(worst).toBeLessThan(TOL_PROBA[model])
        expect(reconWorst).toBeLessThan(model === 'naive_bayes' ? 0.05 : 5e-3)
      })
    }
  }

  it('stale results are rejected after the input or the model changes', async () => {
    const dsId = manifest!.default_dataset
    const ds = manifest!.datasets[dsId]
    const vectors = readJson<TestVectors>(ds.test_vectors)
    const [a, b] = [toPixels01(vectors.pixels[0]), toPixels01(vectors.pixels[1])]
    const ex = await registry!.execute(dsId, 'softmax', a)
    expect(matches(ex, manifest!, dsId, 'softmax', a)).toBe(true)
    expect(matches(ex, manifest!, dsId, 'softmax', b)).toBe(false) // new drawing
    expect(matches(ex, manifest!, dsId, 'mlp', a)).toBe(false) // other model
    expect(matches(ex, manifest!, dsId, 'softmax', null)).toBe(false)
    const other = Object.keys(manifest!.datasets).find((d) => d !== dsId)
    if (other) expect(matches(ex, manifest!, other, 'softmax', a)).toBe(false) // other dataset
  })

  it('refuses a file whose hash differs from the manifest', async () => {
    const dsId = manifest!.default_dataset
    const reg = makeRegistry(manifest!, async (url, init) => {
      const res = await fileFetch(url, init)
      if (!String(url).endsWith('softmax.onnx')) return res
      const b = new Uint8Array(await res.arrayBuffer())
      b[b.length - 1] ^= 1
      return new Response(b)
    })
    await expect(reg.load(dsId, 'softmax')).rejects.toMatchObject({ key: 'err.hash' })
    expect(reg.status(dsId, 'softmax')).toMatchObject({ state: 'error', key: 'err.hash' })
  })
})

describe.skipIf(!hasBundle)('AdaBoost routes and votes', () => {
  it('cumulative votes end at the final scores, leaders follow them, routes are real tree paths', async () => {
    const manifest = loadManifest()
    const registry = makeRegistry(manifest)
    for (const [dsId, ds] of Object.entries(manifest.datasets)) {
      if (!ds.models.adaboost) continue
      const vectors = readJson<TestVectors>(ds.test_vectors)
      const lm = await registry.load(dsId, 'adaboost')
      const ens = lm.ensemble!
      const ex = await registry.execute(dsId, 'adaboost', toPixels01(vectors.pixels[3]))
      const run = ex.ada!
      expect(run.rounds.length).toBe(ens.trees.length)
      // the manifest trace and the file agree on the number of rounds and nodes
      const layer = lm.trace.layers.find((l) => l.op === 'tree_ensemble')!
      expect(layer.n_estimators).toBe(ens.trees.length)
      expect(layer.node_counts).toEqual(ens.nodeCounts)
      let sum = 0
      for (const r of run.rounds) sum += r.alpha
      expect(run.scores.reduce((a, b) => a + b, 0)).toBeCloseTo(sum, 9)
      expect(run.top).toBe(vectors.expected.adaboost.pred[3])
      for (let t = 1; t <= run.rounds.length; t++) {
        const k = run.rounds[t - 1].klass
        expect(run.cumulative[t][k] - run.cumulative[t - 1][k]).toBeCloseTo(run.rounds[t - 1].alpha, 12)
        let best = 0
        for (let c = 1; c < run.cumulative[t].length; c++) if (run.cumulative[t][c] > run.cumulative[t][best]) best = c
        expect(run.leader[t]).toBe(best)
      }
      // every step goes to the child its comparison selects and ends in a leaf of the voted class
      const tree = ens.trees[0]
      const r0 = run.rounds[0]
      for (const s of r0.path) expect(s.goLeft).toBe(s.xStd < tree.t[s.node])
      expect(tree.f[r0.leaf]).toBe(-1)
      expect(tree.c[r0.leaf]).toBe(r0.klass)
      const sizes = subtreeSizes(ens, 0)
      expect(sizes[0]).toBe(ens.nodeCounts[0])
      // the vote shares are normalised votes and the model says they are not calibrated
      expect(lm.trace.output.calibrated).toBe(false)
    }
  })
})

function reconstructionError(model: ModelKind, lm: Awaited<ReturnType<NonNullable<ReturnType<typeof makeRegistry>>['load']>>, ex: Awaited<ReturnType<ReturnType<typeof makeRegistry>['execute']>>): number {
  const o = ex.outputs
  const p = lm.params
  let worst = 0
  const upd = (a: number, b: number) => (worst = Math.max(worst, Math.abs(a - b)))
  const dense = lm.trace.layers.filter((l) => l.op === 'dense')
  if (model === 'mlp' || model === 'softmax') {
    let a: ArrayLike<number> = o.x
    dense.forEach((l, i) => {
      const W = p!.get(l.params!.weight)
      const b = p!.get(l.params!.bias)
      const z = R.denseForward(a, W, b, l.in_features!, l.out_features!)
      const ref = o[l.runtime!]
      for (let j = 0; j < z.length; j++) upd(z[j], ref[j])
      const j = (i * 7 + 3) % z.length // one neuron, term by term
      const n = R.denseNeuron(a, W, b, l.in_features!, l.out_features!, j)
      upd(n.positive + n.negative + n.bias, ref[j])
      a = i < dense.length - 1 ? Float64Array.from(z, (v) => Math.max(v, 0)) : z
    })
    const sm = R.softmax(o.logits)
    for (let k = 0; k < sm.length; k++) upd(sm[k], ex.probs[k])
  } else if (model === 'naive_bayes') {
    const layer = lm.trace.layers.find((l) => l.op === 'gaussian_loglik')!
    const mean = p!.get(layer.params!.mean)
    const variance = p!.get(layer.params!.variance)
    const lp = p!.get(layer.params!.log_prior)
    const joint = Array.from({ length: layer.n_classes! }, (_, k) => R.nbClassTerms(o.x, mean, variance, lp, k).logJoint)
    joint.forEach((v, k) => upd(v, o.logits[k]))
    const post = R.softmax(joint)
    post.forEach((v, k) => upd(v, ex.probs[k]))
  } else if (model === 'cnn') {
    const convs = lm.trace.layers.filter((l) => l.op === 'conv2d')
    let input: ArrayLike<number> = o.x
    let inShape = [1, 28, 28]
    for (const c of convs) {
      const K = p!.get(c.params!.kernel)
      const bias = p!.get(c.params!.bias)
      const conv = o[c.runtime!]
      const [cout, ho, wo] = c.out_shape
      for (const [ch, r, q] of [[0, 0, 0], [cout - 1, ho - 1, wo - 1], [Math.min(1, cout - 1), 2, 3]]) {
        const cp = R.convPatch(input, inShape, K, c.kernel_shape!, bias, ch, r, q, c.stride ?? 1)
        upd(cp.z, conv[(ch * ho + r) * wo + q])
      }
      const reluId = lm.trace.layers.find((l) => l.inputs[0] === c.id && l.op === 'relu')!
      const poolId = lm.trace.layers.find((l) => l.inputs[0] === reluId.id && l.op === 'maxpool2d')!
      const act = o[reluId.runtime!]
      const pool = o[poolId.runtime!]
      const [pc, ph, pw] = poolId.out_shape
      for (const [ch, r, q] of [[0, 0, 0], [pc - 1, ph - 1, pw - 1], [Math.min(1, pc - 1), 1, 2]]) {
        const w = R.poolWindow(act, reluId.out_shape, ch, r, q, poolId.window!, poolId.stride ?? poolId.window)
        upd(w.max, pool[(ch * ph + r) * pw + q])
      }
      input = pool
      inShape = poolId.out_shape
    }
    let a: ArrayLike<number> = input
    dense.forEach((l, i) => {
      const z = R.denseForward(a, p!.get(l.params!.weight), p!.get(l.params!.bias), l.in_features!, l.out_features!)
      const ref = o[l.runtime!]
      for (let j = 0; j < z.length; j++) upd(z[j], ref[j])
      a = i < dense.length - 1 ? Float64Array.from(z, (v) => Math.max(v, 0)) : z
    })
  }
  return worst
}

describe('receptive field arithmetic', () => {
  it('matches hand computation for conv3-pool2-conv3-pool2', () => {
    const stack = [
      { kind: 'conv2d', size: 3, stride: 1 },
      { kind: 'maxpool2d', size: 2, stride: 2 },
      { kind: 'conv2d', size: 3, stride: 1 },
      { kind: 'maxpool2d', size: 2, stride: 2 },
    ] as const
    expect(R.receptiveField(stack.slice(0, 1) as never, 5, 7)).toEqual({ y0: 5, y1: 8, x0: 7, x1: 10 })
    expect(R.receptiveField(stack.slice(0, 2) as never, 0, 0)).toEqual({ y0: 0, y1: 4, x0: 0, x1: 4 })
    // output (1,2) of the second pool covers rows 2*2..2*2+... : conv2 window rows 2..5 -> pool1 rows 4..11 -> input rows 4..13
    const rf = R.receptiveField(stack as never, 1, 2)
    expect(rf).toEqual({ y0: 4, y1: 14, x0: 8, x1: 18 })
  })
})
