import { describe, expect, it } from 'vitest'
import { MODEL_ORDER } from '../src/engine/contract'
import {
  augRows,
  augStressKeys,
  augVerdict,
  curvePanel,
  datasetMatrix,
  datasetSpreads,
  historyPanels,
  isValidMatrix,
  sanitizeResults,
  sweepPanels,
  topConfusions,
} from '../src/lib/results'
import type { AugStudy, Results } from '../src/lib/types'
import { hasBundle, loadManifest, readJson } from './bundle'

describe.skipIf(!hasBundle)('results of the local bundle', () => {
  const manifest = hasBundle ? loadManifest() : null
  const all = hasBundle ? Object.fromEntries(Object.entries(manifest!.datasets).map(([id, ds]) => [id, sanitizeResults(readJson(ds.results))])) : {}

  it('describes the models of every dataset with square confusion matrices of equal totals', () => {
    for (const [id, r] of Object.entries(all)) {
      expect(r.models.length, id).toBeGreaterThan(0)
      for (const s of ['train', 'val', 'test'] as const) {
        const totals = r.models.map((m) => {
          const cm = m.splits?.[s]?.confusion_matrix
          expect(isValidMatrix(cm)).toBe(true)
          expect(cm!.length).toBe(manifest!.datasets[id].classes.length)
          return cm!.flat().reduce((a, b) => a + b, 0)
        })
        expect(new Set(totals).size).toBe(1)
      }
    }
  })

  it('builds history panels of the iterative models', () => {
    const r = all[manifest!.default_dataset]
    const mlp = r.models.find((m) => m.id === 'mlp')
    if (mlp) expect(historyPanels('mlp', mlp.history).map((p) => p.id)).toContain('loss')
    expect(historyPanels('naive_bayes', null)).toEqual([])
    const t = r.tuning?.adaboost
    if (t?.curve) expect(curvePanel(t)?.val.length).toBeGreaterThan(1)
    expect(curvePanel(r.tuning?.naive_bayes)).toBeNull()
  })
})

const run = (seed: number, val: number, rob: Record<string, number>) => ({
  seed,
  n_train: 100,
  fit_time_sec: 1,
  train_acc: 0.99,
  val_acc: val,
  test_acc: val - 0.001,
  val_f1_macro: val,
  test_f1_macro: val,
  robust_test_acc: rob,
})

/** Shape of `src/experiments/augment.py::run_ablation`; the numbers are made up for the test only. */
const study: AugStudy = {
  schema: 2,
  model: 'cnn',
  params: {},
  k_copies: 1,
  seeds: [1, 2, 3],
  defaults: { rotation: 15, scaling: [0.9, 1.1] },
  stress: { rotation: 20, scaling: [0.8, 1.2] },
  selection_rule: 'mean validation accuracy over seeds',
  selected_by_validation: 'rotation',
  conditions: {
    baseline: {
      ops: [],
      runs: [run(1, 0.98, { rotation: 0.9, scaling: 0.95 }), run(2, 0.981, {}), run(3, 0.979, {})],
      train_acc_mean: 0.99,
      train_acc_std: 0,
      val_acc_mean: 0.98,
      val_acc_std: 0.001,
      test_acc_mean: 0.979,
      test_acc_std: 0.001,
      robust_test_acc_mean: { rotation: 0.9, scaling: 0.95 },
      val_delta_vs_baseline_mean: 0,
      val_delta_vs_baseline_sd: 0,
      augmentation_time_sec: 0,
    },
    rotation: {
      ops: ['rotation'],
      runs: [run(1, 0.985, {}), run(2, 0.986, {}), run(3, 0.984, {})],
      train_acc_mean: 0.99,
      train_acc_std: 0,
      val_acc_mean: 0.985,
      val_acc_std: 0.001,
      test_acc_mean: 0.984,
      test_acc_std: 0.001,
      robust_test_acc_mean: { rotation: 0.95, scaling: 0.96 },
      val_delta_vs_baseline_mean: 0.005,
      val_delta_vs_baseline_sd: 0.001,
      augmentation_time_sec: 2,
    },
    salt_pepper: {
      ops: ['salt_pepper'],
      runs: [run(1, 0.98, {}), run(2, 0.979, {}), run(3, 0.982, {})],
      train_acc_mean: 0.99,
      train_acc_std: 0,
      val_acc_mean: 0.9803,
      val_acc_std: 0.0015,
      test_acc_mean: 0.979,
      test_acc_std: 0.001,
      robust_test_acc_mean: { rotation: 0.9, scaling: 0.95 },
      val_delta_vs_baseline_mean: 0.0003,
      val_delta_vs_baseline_sd: 0.0012,
      augmentation_time_sec: 1,
    },
  },
}

describe('optional results sections', () => {
  it('keeps well-formed sections and drops malformed ones', () => {
    const raw = {
      models: [{ id: 'mlp', name: 'MLP' }, null, { name: 'no id' }],
      augmentation: { cnn: study, mlp: { conditions: {} } },
      cnn_vs_mlp: { rows: [] },
      cross_eval: [{ trained_on: 'mnist', rows: [{ model: 'cnn', accuracy: 0.1 }], note: 'n' }, { trained_on: 3 }],
    }
    const r = sanitizeResults(raw)
    expect(r.models.map((m) => m.id)).toEqual(['mlp'])
    expect(r.augmentation?.cnn).toBe(study)
    expect(r.augmentation?.mlp).toBeUndefined()
    expect(r.cnn_vs_mlp).toBeUndefined()
    expect(r.cross_eval).toHaveLength(1)
    expect(sanitizeResults({ models: [] })).toEqual({ models: [] })
    expect(sanitizeResults(null).models).toEqual([])
  })

  it('judges augmentation by the paired difference against its spread', () => {
    expect(augVerdict(0.005, 0.001, 3)).toBe('helps')
    expect(augVerdict(-0.005, 0.001, 3)).toBe('hurts')
    expect(augVerdict(0.0003, 0.0012, 3)).toBe('unclear')
    expect(augVerdict(0.005, 0, 1)).toBe('unclear') // one seed: no spread to compare with
    expect(augVerdict(null, 0.001, 3)).toBe('unclear')
    const rows = augRows(study)
    expect(rows.map((r) => [r.name, r.verdict, r.selected])).toEqual([
      ['baseline', 'baseline', false],
      ['rotation', 'helps', true],
      ['salt_pepper', 'unclear', false],
    ])
    expect(augStressKeys(study)).toEqual(['rotation', 'scaling'])
  })

  it('builds the dataset comparison from whatever results exist', () => {
    const a: Results = { models: [{ id: 'softmax', name: 's', splits: { test: { accuracy: 0.92, f1_macro: 0.91 } } }, { id: 'cnn', name: 'c', splits: { test: { accuracy: 0.99, f1_macro: 0.99 } } }] }
    const b: Results = { models: [{ id: 'cnn', name: 'c', splits: { test: { accuracy: 0.9 } } }] }
    const m = datasetMatrix({ mnist: a, fashion_mnist: b }, ['mnist', 'fashion_mnist', 'emnist_digits'], MODEL_ORDER)
    expect(m.models).toEqual(['softmax', 'cnn'])
    expect(m.cells.fashion_mnist.cnn).toEqual({ accuracy: 0.9, f1: null })
    const s = datasetSpreads(m)
    expect(s[0].best?.model).toBe('cnn')
    expect(s[0].gapPoints).toBeCloseTo(7, 10)
    expect(s[1].gapPoints).toBeNull()
    expect(s[2].best).toBeNull()
  })
})

describe('results helpers', () => {
  it('finds the largest off-diagonal cells', () => {
    const cm = [
      [90, 5, 0],
      [1, 80, 9],
      [0, 3, 70],
    ]
    const top = topConfusions(cm, 2)
    expect(top[0]).toMatchObject({ trueDigit: 1, predDigit: 2, count: 9 })
    expect(top[1]).toMatchObject({ trueDigit: 0, predDigit: 1, count: 5 })
    expect(top[0].share).toBeCloseTo(9 / 90, 10)
    expect(topConfusions(null)).toEqual([])
  })

  it('rejects malformed matrices', () => {
    expect(isValidMatrix([[1, 2], [3]])).toBe(false)
    expect(isValidMatrix(null)).toBe(false)
  })

  it('builds history panels, tolerating models without history', () => {
    expect(historyPanels('mlp', { loss: [1, 0.5], val_loss: [1.1, 0.6], inner_val_loss: [1, 0.7], val_acc: [0.9, 0.95] }).map((p) => p.id)).toEqual(['loss', 'acc'])
    expect(historyPanels('adaboost', { train_err_weighted: [0.5, 0.4], val_acc: [0.8, 0.9], alpha: [1, 0.9] }).map((p) => p.id)).toEqual(['loss', 'acc', 'alpha'])
  })

  it('ignores unknown history series', () => {
    const panels = historyPanels('mlp', { loss: [1, 0.5], val_loss: [1.1, 0.6], inner_val_loss: [1, 0.7], made_up: [1, 2], val_acc: [Number.NaN] })
    expect(panels.map((p) => p.id)).toEqual(['loss'])
    expect(panels[0].series.map((s) => s.key).sort()).toEqual(['inner_val_loss', 'loss', 'val_loss'])
    expect(historyPanels('naive_bayes', null)).toEqual([])
  })

  it('builds sweep panels with the chosen value, ignoring single-point sweeps', () => {
    const panels = sweepPanels({
      best_params: { hidden_sizes: [32], learning_rate: 0.03, epochs: 6 },
      sweeps: {
        hidden_sizes: [
          { value: [16], train_acc: 0.8, val_acc: 0.78 },
          { value: [32], train_acc: 0.93, val_acc: 0.88 },
        ],
        learning_rate: [
          { value: 0.3, train_acc: 0.9, val_acc: 0.85 },
          { value: 0.003, train_acc: 0.8, val_acc: 0.79 },
          { value: 0.03, train_acc: 0.92, val_acc: 0.88 },
        ],
        epochs: [{ value: 6, train_acc: 0.9, val_acc: 0.88 }],
      },
    })
    expect(panels.map((p) => p.param)).toEqual(['hidden_sizes', 'learning_rate'])
    expect(panels[0]).toMatchObject({ kind: 'categorical', chosen: { x: 1, label: '[32]' } })
    expect(panels[1]).toMatchObject({ kind: 'numeric', log: true, chosen: { x: 0.03 } })
    expect(panels[1].points.map((p) => p.x)).toEqual([0.003, 0.03, 0.3])
    expect(sweepPanels(null)).toEqual([])
  })

})
