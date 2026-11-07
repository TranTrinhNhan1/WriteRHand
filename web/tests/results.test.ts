import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { curvePanel, historyPanels, isValidMatrix, sweepPanels, topConfusions } from '../src/lib/results'
import type { Manifest, Results } from '../src/lib/types'

const dataDir = resolve(__dirname, '../public/data')
const results = JSON.parse(readFileSync(resolve(dataDir, 'results.json'), 'utf8')) as Results
const manifest = JSON.parse(readFileSync(resolve(dataDir, 'manifest.json'), 'utf8')) as Manifest

describe('committed site data', () => {
  it('describes the four browser models in both files', () => {
    expect(results.models.map((m) => m.id).sort()).toEqual(['adaboost', 'mlp', 'naive_bayes', 'softmax'])
    expect(Object.keys(manifest.models).sort()).toEqual(['adaboost', 'mlp', 'naive_bayes', 'softmax'])
  })

  it('has square confusion matrices with the same number of images for every model', () => {
    for (const s of ['train', 'val', 'test'] as const) {
      const totals = results.models.map((m) => {
        const cm = m.splits?.[s]?.confusion_matrix
        expect(isValidMatrix(cm)).toBe(true)
        return cm!.flat().reduce((a, b) => a + b, 0)
      })
      expect(new Set(totals).size).toBe(1)
      expect(totals[0]).toBeGreaterThan(0)
    }
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
    const mlp = results.models.find((m) => m.id === 'mlp')!
    const panels = historyPanels('mlp', mlp.history)
    expect(panels.map((p) => p.id)).toEqual(['loss', 'acc'])
    expect(panels[0].series.map((s) => s.key).sort()).toEqual(['inner_val_loss', 'loss', 'val_loss'])
    expect(historyPanels('naive_bayes', null)).toEqual([])
    const ada = historyPanels('adaboost', results.models.find((m) => m.id === 'adaboost')!.history)
    expect(ada.map((p) => p.id)).toEqual(['loss', 'acc', 'alpha'])
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
    expect(panels[0]).toMatchObject({ kind: 'categorical', chosen: { x: 1, label: '32' } })
    expect(panels[1]).toMatchObject({ kind: 'numeric', log: true, chosen: { x: 0.03 } })
    expect(panels[1].points.map((p) => p.x)).toEqual([0.003, 0.03, 0.3])
    expect(sweepPanels(null)).toEqual([])
  })

  it('reads the tuning curve', () => {
    const t = results.tuning?.adaboost
    const c = curvePanel(t)
    expect(c?.param).toBe('n_estimators')
    expect(c?.val.length).toBeGreaterThan(1)
    expect(curvePanel(results.tuning?.naive_bayes)).toBeNull()
  })
})
