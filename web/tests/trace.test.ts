import { describe, expect, it } from 'vitest'
import * as ort from 'onnxruntime-web'
import { adaboostPredict, argmax, parseAdaBoostSpec, type AdaBoostSpec } from '../src/lib/adaboost'
import { createOnnxPredictor, INPUT_DIM } from '../src/lib/onnx'
import { TRACE_OUTPUTS, adaboostTrace, N_PIXELS } from '../src/lib/trace'
import type { Manifest } from '../src/lib/types'
import { hasBundle, readBytes, readJson, type TestVectors } from './helpers'

function softmax(z: ArrayLike<number>): number[] {
  const m = Math.max(...Array.from(z))
  const e = Array.from(z, (v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

describe('AdaBoost trace', () => {
  // two trees on 784 features: tree 0 splits pixel 3, tree 1 is a single leaf
  const spec: AdaBoostSpec = {
    type: 'adaboost-samme',
    classes: [0, 1, 2],
    alphas: [0.7, 0.3],
    preprocess: { mean: 0.1, std: 0.3, eps: 0 },
    trees: [
      { f: [3, -1, -1], t: [0.5, 0, 0], l: [1, -1, -1], r: [2, -1, -1], c: [0, 1, 2] },
      { f: [-1], t: [0], l: [-1], r: [-1], c: [2] },
    ],
  }

  it('records the path, the leaf and the weight of every tree, in raw pixel units', () => {
    const px = new Float32Array(N_PIXELS)
    px[3] = 1
    const tr = adaboostTrace(spec, px)
    expect(tr.rounds).toHaveLength(2)
    const first = tr.rounds[0]
    expect(first.path).toHaveLength(1)
    expect(first.path[0].feature).toBe(3)
    expect(first.path[0].goLeft).toBe(false)
    expect(first.path[0].value).toBe(1)
    expect(first.path[0].threshold).toBeCloseTo(0.5 * 0.3 + 0.1, 12)
    expect(first.path[0].otherSize).toBe(1)
    expect(first.digit).toBe(2)
    expect(tr.rounds[1].path).toHaveLength(0)
  })

  it('adds up to the same vote shares as the predictor', () => {
    const px = new Float32Array(N_PIXELS).fill(0.2)
    const tr = adaboostTrace(spec, px)
    const sums = [0, 0, 0]
    for (const r of tr.rounds) sums[spec.classes.indexOf(r.digit)] += r.alpha
    const total = sums.reduce((a, b) => a + b, 0)
    const p = adaboostPredict(spec, px)
    sums.forEach((s, k) => expect(s / total).toBeCloseTo(p[k], 12))
  })
})

describe.skipIf(!hasBundle)('trace outputs of the exported models', () => {
  const vectors = hasBundle ? readJson<TestVectors>('test_vectors.json') : null
  const manifest = hasBundle ? readJson<Manifest>('manifest.json') : null
  const px = () => Float32Array.from(vectors!.pixels[0].map((v) => v / 255))

  for (const id of ['naive_bayes', 'softmax', 'mlp'] as const) {
    it(`${id}: the traced scores turn into the model's probabilities`, async () => {
      const entry = manifest!.models[id]!
      const predictor = await createOnnxPredictor(ort as never, readBytes(entry.file))
      const probs = await predictor.run(px(), 1)
      const t = await predictor.trace(px(), TRACE_OUTPUTS[id]!)
      expect(t).not.toBeNull()
      const sm = softmax(t!.logits)
      sm.forEach((v, k) => expect(Math.abs(v - probs[k])).toBeLessThan(1e-5))
      if (id === 'mlp') {
        expect(t!.a0.length).toBe(512)
        expect(t!.a1.length).toBe(256)
        expect(Math.min(...Array.from(t!.a0))).toBeGreaterThanOrEqual(0)
      }
      if (id === 'softmax') {
        for (let c = 0; c < 10; c++) {
          let s = t!.bias[c]
          for (let j = 0; j < INPUT_DIM; j++) s += t!.x[j] * t!.weights[j * 10 + c]
          expect(Math.abs(s - t!.logits[c])).toBeLessThan(1e-3)
        }
      }
      if (id === 'naive_bayes') {
        // what the "look inside" view recomputes in the browser must equal the model's own scores
        for (let c = 0; c < 10; c++) {
          let s = t!.log_prior[c]
          for (let j = 0; j < INPUT_DIM; j++) {
            const v = t!.class_var[c * INPUT_DIM + j]
            const d = t!.x[j] - t!.class_mean[c * INPUT_DIM + j]
            s += -0.5 * (Math.log(2 * Math.PI * v) + (d * d) / v)
          }
          expect(Math.abs(s - t!.logits[c])).toBeLessThan(0.05)
        }
      }
      await predictor.dispose()
    })
  }

  it('adaboost: the traced path ends in the leaf the predictor uses', () => {
    const spec = parseAdaBoostSpec(readJson('models/adaboost.json'))
    const tr = adaboostTrace(spec, px())
    const sums = new Array(10).fill(0)
    for (const r of tr.rounds) sums[tr.classes.indexOf(r.digit)] += r.alpha
    expect(argmax(sums)).toBe(argmax(adaboostPredict(spec, px())))
  })
})
