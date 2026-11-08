import { describe, expect, it } from 'vitest'
import type { BundleManifest, DatasetEntry, ModelEntry } from '../src/engine/contract'
import { axisLabel } from '../src/components/charts/ConfusionMatrix'
import { parameterCount, summarizeLayers } from '../src/lib/architecture'
import { AUTO_LOAD_MAX_BYTES, consensusOf, datasetIdsOf, isDigitDataset, modelChoices, modelPairs, resolveDataset, resolveModel, shortClassNames, variantId } from '../src/lib/selection'
import { modelsOf } from '../src/engine/contract'
import { parseTestVectors } from '../src/lib/testImages'
import { hasBundle, loadManifest } from './bundle'

/** Minimal manifest entries for the selection rules; only the fields those rules read are real. */
const entry = (ok: boolean, size: number, nbytes = 0): ModelEntry =>
  ({ size_bytes: size, validation: { ok, max_abs_prob_diff: 0, argmax_agreement: 1, n_checked: 1 }, trace: { params: nbytes ? { nbytes } : null } }) as unknown as ModelEntry

const fashion = {
  title: 'Fashion-MNIST',
  kind: 'clothing',
  classes: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
  models: { adaboost: entry(true, 9_000_000), naive_bayes: entry(true, 60_000, 60_000), cnn: entry(false, 20_000, 20_000), mlp: entry(true, 2_000_000, 2_000_000) },
} as unknown as DatasetEntry

describe('dataset and model choice', () => {
  it('lists the models of a dataset in the fixed order, with the loading policy', () => {
    const c = modelChoices(fashion)
    expect(c.map((x) => x.id)).toEqual(['naive_bayes', 'mlp', 'cnn', 'adaboost'])
    expect(c.find((x) => x.id === 'cnn')).toMatchObject({ selectable: false, autoLoad: false })
    expect(c.find((x) => x.id === 'adaboost')).toMatchObject({ selectable: true, autoLoad: false })
    expect(c.find((x) => x.id === 'mlp')).toMatchObject({ selectable: true, autoLoad: true, bytes: 4_000_000 })
    expect(4_000_000).toBeLessThanOrEqual(AUTO_LOAD_MAX_BYTES)
  })

  it('never selects a model that failed its export check or is missing', () => {
    expect(resolveModel(fashion, 'cnn')).toBe('mlp') // cnn failed: falls back to the next preferred
    expect(resolveModel(fashion, 'softmax')).toBe('mlp') // not in this dataset
    expect(resolveModel(fashion, 'adaboost')).toBe('adaboost')
    expect(resolveModel({ ...fashion, models: { cnn: entry(false, 1) } } as DatasetEntry, null)).toBeNull()
  })

  it('falls back to the default dataset and puts it first', () => {
    const m = { default_dataset: 'mnist', datasets: { fashion_mnist: fashion, mnist: { ...fashion, kind: 'digits' } } } as unknown as BundleManifest
    expect(datasetIdsOf(m)).toEqual(['mnist', 'fashion_mnist'])
    expect(resolveDataset(m, 'emnist_digits')).toBe('mnist')
    expect(resolveDataset(m, 'fashion_mnist')).toBe('fashion_mnist')
    expect(isDigitDataset(fashion)).toBe(false)
    expect(shortClassNames(fashion.classes)).toBe(false)
    expect(shortClassNames(['0', '1'])).toBe(true)
  })
})

describe('consensus', () => {
  it('takes the majority and breaks ties by the mean top score', () => {
    expect(consensusOf([])).toBeNull()
    expect(consensusOf([{ model: 'mlp', top: 7, score: 0.9 }])).toEqual({ cls: 7, votes: 1, of: 1, tied: false })
    const v = consensusOf([
      { model: 'mlp', top: 7, score: 0.9 },
      { model: 'cnn', top: 7, score: 0.8 },
      { model: 'softmax', top: 1, score: 0.99 },
    ])
    expect(v).toEqual({ cls: 7, votes: 2, of: 3, tied: false })
    const tie = consensusOf([
      { model: 'mlp', top: 3, score: 0.6 },
      { model: 'cnn', top: 5, score: 0.7 },
    ])
    expect(tie).toEqual({ cls: 5, votes: 1, of: 2, tied: true })
  })
})

describe('test images', () => {
  it('scales pixels to [0, 1], orders by label and rejects bad files', () => {
    const px = (v: number) => Array.from({ length: 784 }, () => v)
    const imgs = parseTestVectors({ indices: [9, 4], labels: [3, 1], pixels: [px(255), px(51)] }, 10)
    expect(imgs.map((i) => [i.index, i.label])).toEqual([
      [4, 1],
      [9, 3],
    ])
    expect(imgs[0].pixels[0]).toBeCloseTo(0.2, 6)
    expect(imgs[1].pixels[783]).toBe(1)
    expect(() => parseTestVectors({ indices: [1], labels: [10], pixels: [px(0)] }, 10)).toThrow()
    expect(() => parseTestVectors({ indices: [1], labels: [1], pixels: [[1, 2]] }, 10)).toThrow()
    expect(() => parseTestVectors(null, 10)).toThrow()
  })
})

describe('labels', () => {
  it('shortens long class names for chart axes only', () => {
    expect(axisLabel('Bag', 5)).toBe('Bag')
    expect(axisLabel('Ankle boot', 8)).toBe('Ankle b…')
    expect(axisLabel('T-shirt/top', 8)).toBe('T-shirt…')
  })
})

describe.skipIf(!hasBundle)('architecture summary of the local bundle', () => {
  it('is read from the trace and counts exactly the parameters of the index', () => {
    const m = loadManifest()
    for (const ds of Object.values(m.datasets)) {
      for (const e of Object.values(ds.models)) {
        const layers = summarizeLayers(e!.trace)
        expect(layers.map((l) => l.id)).toEqual(e!.trace.layers.map((l) => l.id))
        if (e!.trace.params) {
          const sum = layers.reduce((a, l) => a + (l.params ?? 0), 0)
          expect(sum).toBe(parameterCount(e!.trace))
        }
        const conv = e!.trace.layers.find((l) => l.op === 'conv2d')
        if (conv) expect(layers.find((l) => l.id === conv.id)!.detail).toContain(`${conv.kernel_shape![2]}×${conv.kernel_shape![3]}`)
      }
    }
  })
})

describe.skipIf(!hasBundle)('plain / augmented pairs of the local bundle', () => {
  it('lists one row per architecture, augmented variants never load by themselves', () => {
    const m = loadManifest()
    for (const ds of Object.values(m.datasets)) {
      const choices = modelChoices(ds)
      const pairs = modelPairs(choices)
      expect(pairs.map((p) => p.kind)).toEqual(modelsOf(ds))
      for (const c of choices) if (c.augmented) expect(c.autoLoad).toBe(false)
      for (const p of pairs) if (p.augmented) expect(p.augmented.id).toBe(`${p.kind}_aug`)
    }
    const mnist = m.datasets.mnist
    if (mnist?.models.mlp_aug) {
      expect(variantId(mnist, 'mlp', true)).toBe('mlp_aug')
      expect(resolveModel(mnist, 'mlp_aug')).toBe('mlp_aug')
    }
  })
})
