import { describe, expect, it } from 'vitest'
import * as ort from 'onnxruntime-web'
import { adaboostPredict, argmax, parseAdaBoostSpec } from '../src/lib/adaboost'
import { createOnnxPredictor, INPUT_DIM, N_CLASSES } from '../src/lib/onnx'
import { MNIST_SIZE, grayFromRGBA, preprocessMnist, upsample } from '../src/lib/preprocess'
import type { Manifest } from '../src/lib/types'
import { MODEL_IDS } from '../src/lib/types'
import { hasBundle, readBytes, readJson, type TestVectors } from './helpers'

// These tests compare the browser inference code against the numbers the Python pipeline wrote
// to test_vectors.json. They need the model bundle (see README); without it they are skipped.
describe.skipIf(!hasBundle)('browser inference matches Python', () => {
  const vectors = hasBundle ? readJson<TestVectors>('test_vectors.json') : null
  const manifest = hasBundle ? readJson<Manifest>('manifest.json') : null
  const n = vectors?.pixels.length ?? 0

  const pixels01 = (i: number) => vectors!.pixels[i].map((v) => v / 255)
  const flat = () => {
    const out = new Float32Array(n * INPUT_DIM)
    for (let i = 0; i < n; i++) out.set(pixels01(i), i * INPUT_DIM)
    return out
  }

  it('has the expected shape', () => {
    expect(n).toBeGreaterThan(0)
    expect(Object.keys(vectors!.expected).sort()).toEqual([...MODEL_IDS].sort())
  })

  it('AdaBoost evaluator reproduces Python (within the 6-decimal rounding of test_vectors.json) and argmax', () => {
    const spec = parseAdaBoostSpec(readJson('models/adaboost.json'))
    const exp = vectors!.expected.adaboost
    for (let i = 0; i < n; i++) {
      const p = adaboostPredict(spec, pixels01(i))
      expect(argmax(p)).toBe(exp.pred[i])
      for (let k = 0; k < N_CLASSES; k++) expect(Math.abs(p[k] - exp.proba[i][k])).toBeLessThanOrEqual(5.01e-7)
    }
  })

  for (const id of ['naive_bayes', 'softmax', 'mlp'] as const) {
    it(`ONNX ${id} reproduces Python within 1e-3`, async () => {
      const entry = manifest!.models[id]!
      const predictor = await createOnnxPredictor(ort as never, readBytes(entry.file))
      const probs = await predictor.run(flat(), n)
      const exp = vectors!.expected[id]
      let worst = 0
      for (let i = 0; i < n; i++) {
        const row = probs.slice(i * N_CLASSES, (i + 1) * N_CLASSES)
        expect(argmax(row)).toBe(exp.pred[i])
        for (let k = 0; k < N_CLASSES; k++) worst = Math.max(worst, Math.abs(row[k] - exp.proba[i][k]))
      }
      expect(worst).toBeLessThan(1e-3)
      // A single-image call (what the UI does) agrees with the batched call.
      const one = await predictor.run(flat().slice(0, INPUT_DIM), 1)
      expect(Array.from(one)).toEqual(Array.from(probs.slice(0, N_CLASSES)))
      await predictor.dispose()
    })
  }

  it('round trip: a MNIST digit enlarged 10x and preprocessed again is close to the original', async () => {
    const mlpEntry = manifest!.models.mlp!
    const predictor = await createOnnxPredictor(ort as never, readBytes(mlpEntry.file))
    let sameClass = 0
    let sumMae = 0
    for (let i = 0; i < n; i++) {
      const orig = Float32Array.from(pixels01(i))
      const big = upsample(orig, MNIST_SIZE, 10)
      const out = preprocessMnist(big)
      expect(out).not.toBeNull()
      let mae = 0
      for (let j = 0; j < orig.length; j++) mae += Math.abs(orig[j] - out!.pixels[j])
      sumMae += mae / orig.length
      expect(Math.abs(out!.com.x - 14)).toBeLessThan(0.5)
      expect(Math.abs(out!.com.y - 14)).toBeLessThan(0.5)
      const a = argmax(await predictor.run(orig, 1))
      const b = argmax(await predictor.run(out!.pixels, 1))
      if (a === b) sameClass++
    }
    // Mean absolute pixel error stays small (a full-intensity digit covers ~19 % of the image).
    expect(sumMae / n).toBeLessThan(0.06)
    expect(sameClass).toBeGreaterThanOrEqual(Math.ceil(n * 0.75))
    await predictor.dispose()
  })

  it('RGBA helper feeds the same pipeline', () => {
    const rgba = new Uint8ClampedArray(MNIST_SIZE * MNIST_SIZE * 4)
    for (let i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
      const v = vectors!.pixels[0][i]
      rgba.set([v, v, v, 255], i * 4)
    }
    const gray = grayFromRGBA(rgba, MNIST_SIZE, MNIST_SIZE)
    expect(preprocessMnist(upsample(gray.data, MNIST_SIZE, 4))).not.toBeNull()
  })
})
