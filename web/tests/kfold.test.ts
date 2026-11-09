import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { MODEL_ORDER } from '../src/engine/contract'
import { KFOLD_SCHEMA, parseKFold } from '../src/lib/kfold'

const committed = JSON.parse(readFileSync(resolve(__dirname, '../public/data/kfold.json'), 'utf8'))
const source = (name: string) => JSON.parse(readFileSync(resolve(__dirname, '../../results/kfold', `${name}.json`), 'utf8'))

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
const sd = (v: number[]) => {
  const m = mean(v)
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1))
}

describe('kfold.json (k-fold cross-validation, MNIST)', () => {
  it('parses the committed file: six models in display order, k folds each', () => {
    const kf = parseKFold(committed)!
    expect(kf).not.toBeNull()
    expect(kf.k).toBe(committed.k)
    expect(kf.models.map((m) => m.id)).toEqual(MODEL_ORDER)
    for (const m of kf.models) expect(m.folds.map((f) => f.fold)).toEqual(Array.from({ length: kf.k }, (_, i) => i + 1))
  })

  it('has exactly the numbers of results/kfold/<Model>.json, with mean and sd (ddof = 1) over the folds', () => {
    for (const m of committed.models) {
      const r = source(m.name)
      expect(r.k, m.name).toBe(committed.k)
      expect(r.n_pool).toBe(committed.n_pool)
      expect(r.n_test).toBe(committed.n_test)
      const folds = Object.keys(r.folds)
        .sort((a, b) => Number(a) - Number(b))
        .map((k) => r.folds[k])
      expect(folds).toHaveLength(committed.k)
      expect(m.folds).toHaveLength(committed.k)
      folds.forEach((f, i) => {
        const g = m.folds[i]
        expect(g.fold).toBe(i + 1)
        expect(g.n_train).toBe(f.n_train)
        expect(g.n_val).toBe(f.n_val)
        expect(g.n_train + g.n_val).toBe(committed.n_pool)
        expect(g.val_accuracy).toBe(f.val.accuracy)
        expect(g.test_accuracy).toBe(f.test.accuracy)
        expect(g.val_f1_macro).toBe(f.val.f1_macro)
        expect(g.test_f1_macro).toBe(f.test.f1_macro)
      })
      for (const split of ['val', 'test'] as const) {
        const v = folds.map((f) => f[split].accuracy)
        expect(m[`${split}_accuracy`].mean, `${m.name} ${split}`).toBeCloseTo(mean(v), 12)
        expect(m[`${split}_accuracy`].std, `${m.name} ${split}`).toBeCloseTo(sd(v), 12)
      }
    }
    // the parser keeps the numbers as they are
    const kf = parseKFold(committed)!
    for (const m of kf.models) {
      const raw = committed.models.find((x: { id: string }) => x.id === m.id)
      expect(m.val_accuracy).toEqual(raw.val_accuracy)
      expect(m.test_accuracy).toEqual(raw.test_accuracy)
      expect(m.folds.map((f) => f.val_accuracy)).toEqual(raw.folds.map((f: { val_accuracy: number }) => f.val_accuracy))
    }
  })

  it('rejects another schema or shape, and drops malformed or repeated models', () => {
    expect(parseKFold(null)).toBeNull()
    expect(parseKFold({ ...committed, schema: 'other' })).toBeNull()
    expect(parseKFold({ ...committed, k: 1 })).toBeNull()
    expect(parseKFold({ ...committed, models: [] })).toBeNull()
    const [nb, sm] = committed.models
    const shortFolds = { ...sm, folds: sm.folds.slice(1) }
    const badAcc = { ...sm, folds: sm.folds.map((f: object, i: number) => (i ? f : { ...f, val_accuracy: 1.5 })) }
    const unknown = { ...nb, id: 'stacking' }
    for (const bad of [shortFolds, badAcc, unknown]) {
      const r = parseKFold({ schema: KFOLD_SCHEMA, k: committed.k, models: [nb, bad, nb] })!
      expect(r.models.map((m) => m.id)).toEqual(['naive_bayes'])
    }
  })
})
