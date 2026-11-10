import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DATASETS_SCHEMA, IMAGE_PIXELS, decodeImage, parseDatasets } from '../src/lib/datasets'

const read = (rel: string) => JSON.parse(readFileSync(resolve(__dirname, '../public/data', rel), 'utf8'))
const committed = read('datasets.json')
/** the manifest committed with the site: split sizes and class names per dataset */
const manifest = read('manifest.json')
const SPLITS = ['train', 'val', 'test'] as const

describe('datasets.json (dataset introduction on the About page)', () => {
  const parsed = parseDatasets(committed)

  it('parses the committed file: every dataset of the manifest, nothing dropped', () => {
    expect(Object.keys(parsed).sort()).toEqual(Object.keys(committed.datasets).sort())
    expect(Object.keys(parsed).sort()).toEqual(Object.keys(manifest.datasets).sort())
  })

  for (const id of Object.keys(manifest.datasets)) {
    describe(id, () => {
      it('has the ten classes of the manifest, in order, with 8 samples of 784 grey levels and a mean image', () => {
        const d = parsed[id]
        expect(d.title).toBe(manifest.datasets[id].title)
        expect(d.classes.map((c) => c.label)).toEqual(manifest.datasets[id].classes)
        expect(d.classes).toHaveLength(10)
        for (const c of d.classes) {
          expect(c.samples, c.label).toHaveLength(8)
          for (const img of [c.mean, ...c.samples]) {
            expect(atob(img).length).toBe(IMAGE_PIXELS)
            expect(decodeImage(img)).toHaveLength(IMAGE_PIXELS)
          }
          // a mean image of real images is never blank, and no sample image is blank either
          expect(decodeImage(c.mean).some((v) => v > 0), c.label).toBe(true)
          for (const s of c.samples) expect(decodeImage(s).some((v) => v > 0)).toBe(true)
        }
        expect(d.ink_fraction).toBeGreaterThan(0)
        expect(d.ink_fraction).toBeLessThan(1)
      })

      it('has split sizes equal to the manifest, and class counts that add up to them', () => {
        const d = parsed[id]
        const meta = manifest.datasets[id].dataset
        expect(d.n).toEqual({ train: meta.n_train, val: meta.n_val, test: meta.n_test })
        for (const s of SPLITS) expect(d.classes.reduce((a, c) => a + c.n[s], 0), s).toBe(d.n[s])
      })

      it('has per-class counts equal to the row sums of the measured confusion matrices (results.json)', () => {
        const results = read(manifest.datasets[id].results)
        for (const m of results.models) {
          for (const s of SPLITS) {
            const cm: number[][] | undefined = m.splits[s]?.confusion_matrix
            if (!cm) continue
            expect(
              cm.map((row) => row.reduce((a, b) => a + b, 0)),
              `${m.id} ${s}`,
            ).toEqual(parsed[id].classes.map((c) => c.n[s]))
          }
        }
      })
    })
  }
})

describe('parseDatasets', () => {
  const clone = () => JSON.parse(JSON.stringify(committed))

  it('rejects another schema, another image size, or no datasets', () => {
    expect(parseDatasets(null)).toEqual({})
    expect(parseDatasets({ ...clone(), schema: 'writerhand.datasets/2' })).toEqual({})
    expect(parseDatasets({ ...clone(), size: 32 })).toEqual({})
    expect(parseDatasets({ schema: DATASETS_SCHEMA, size: 28 })).toEqual({})
  })

  it('drops only the dataset whose images, counts or fields are malformed', () => {
    const bad = clone()
    bad.datasets.mnist.classes[3].samples[2] = bad.datasets.mnist.classes[3].samples[2].slice(4)
    bad.datasets.fashion_mnist.classes[0].n.train += 1
    bad.datasets.emnist_digits.ink_fraction = 1.5
    expect(Object.keys(parseDatasets(bad))).toEqual([])
    const one = clone()
    one.datasets.mnist.kind = 'letters'
    expect(Object.keys(parseDatasets(one)).sort()).toEqual(['emnist_digits', 'fashion_mnist'])
  })

  it('decodes an image byte for byte', () => {
    const bytes = Uint8Array.from({ length: IMAGE_PIXELS }, (_, i) => (i * 7) % 256)
    const b64 = Buffer.from(bytes).toString('base64')
    expect(Array.from(decodeImage(b64))).toEqual(Array.from(bytes))
    const c = clone()
    c.datasets.mnist.classes[0].mean = b64
    expect(parseDatasets(c).mnist.classes[0].mean).toBe(b64)
  })
})
