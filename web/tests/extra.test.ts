import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { EXTRA_SCHEMA, parseExtraModels } from '../src/lib/extraModels'

const committed = JSON.parse(readFileSync(resolve(__dirname, '../public/data/extra_models.json'), 'utf8'))
const manifest = JSON.parse(readFileSync(resolve(__dirname, '../public/data/manifest.json'), 'utf8'))

describe('extra_models.json (measured models that are not in the browser bundle)', () => {
  it('parses the committed file: linear SVM and stacking on MNIST, never runnable in the browser', () => {
    const extra = parseExtraModels(committed)
    expect(Object.keys(extra)).toEqual(['mnist'])
    const ids = extra.mnist.models.map((m) => m.id)
    expect(ids).toEqual(['svm', 'stacking'])
    for (const m of extra.mnist.models) {
      expect(m.in_browser).toBe(false)
      for (const s of ['train', 'val', 'test'] as const) expect(m.splits?.[s]?.confusion_matrix?.length).toBe(10)
    }
    expect(extra.mnist.models[0].splits?.test?.accuracy).toBeCloseTo(0.9195, 6)
    expect(extra.mnist.models[1].splits?.test?.accuracy).toBeCloseTo(0.9905, 6)
    expect(extra.mnist.models[1].stacking?.fitted_on).toBe('val')
    expect(extra.mnist.models[1].stacking?.bases).toHaveLength(6)
    expect(Object.keys(extra.mnist.tuning).sort()).toEqual(['stacking', 'svm'])
  })

  it('names datasets of the pinned manifest only, and neither model is in the bundle', () => {
    for (const [ds, v] of Object.entries(parseExtraModels(committed))) {
      expect(manifest.datasets[ds], ds).toBeTruthy()
      for (const m of v.models) expect(manifest.datasets[ds].models[m.id], m.id).toBeUndefined()
    }
  })

  it('carries the plain / augmented pairs of MNIST with their measured numbers', () => {
    const pairs = parseExtraModels(committed).mnist.pairs!
    expect(Object.keys(pairs.models)).toEqual(['naive_bayes', 'softmax', 'mlp', 'cnn', 'adaboost'])
    // augmentation does not help every model: Naive Bayes loses accuracy on the clean test set
    expect(pairs.models.naive_bayes!.augmented.test_acc).toBeLessThan(pairs.models.naive_bayes!.plain.test_acc)
    expect(pairs.models.cnn!.augmented.test_acc).toBeGreaterThan(pairs.models.cnn!.plain.test_acc)
    for (const p of Object.values(pairs.models)) for (const a of pairs.rotation_angles) expect(p!.plain.test_rotation_acc[String(a)]).toBeGreaterThan(0)
    expect(pairs.recipe?.ops).toEqual(['rotation', 'translation', 'scaling'])
    // every pair has both models in the pinned manifest
    for (const k of Object.keys(pairs.models)) {
      expect(manifest.datasets.mnist.models[k]).toBeTruthy()
      expect(manifest.datasets.mnist.models[`${k}_aug`]?.augmented).toBe(true)
    }
  })

  it('drops anything with another schema or shape', () => {
    expect(parseExtraModels(null)).toEqual({})
    expect(parseExtraModels({ ...committed, schema: 'other' })).toEqual({})
    const bad = { schema: EXTRA_SCHEMA, datasets: { mnist: { models: [{ id: 'cnn', splits: { test: { accuracy: 1 } } }, { id: 'stacking', splits: { test: { accuracy: 1 } } }] } } }
    expect(parseExtraModels(bad)).toEqual({})
    const ok = { schema: EXTRA_SCHEMA, datasets: { mnist: { models: [{ id: 'svm', in_browser: true, splits: { test: { accuracy: 0.9 } } }], tuning: { stacking: {} } } } }
    const r = parseExtraModels(ok)
    expect(r.mnist.models[0].in_browser).toBe(false)
    expect(r.mnist.tuning).toEqual({})
  })
})
