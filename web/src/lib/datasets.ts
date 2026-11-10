import { finite } from './scale'

/** `public/data/datasets.json`, written by `python -m src site-extra` (src/experiments/site_datasets.py). */
export const DATASETS_SCHEMA = 'writerhand.datasets/1'
export const IMAGE_SIZE = 28
export const IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
/** base64 length of one image (784 bytes, padded) */
const B64_LEN = Math.ceil(IMAGE_PIXELS / 3) * 4
const B64 = /^[A-Za-z0-9+/]+={0,2}$/

export interface SplitCounts {
  train: number
  val: number
  test: number
}

export interface DatasetClass {
  label: string
  n: SplitCounts
  /** base64 of 784 grey levels (0 = empty, 255 = full ink): the mean of the class's training images */
  mean: string
  /** base64 training images of this class, 784 grey levels each */
  samples: string[]
}

export interface DatasetIntro {
  title: string
  kind: 'digits' | 'clothing'
  n: SplitCounts
  /** share of training pixels that are not empty (> 0) */
  ink_fraction: number
  classes: DatasetClass[]
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)
const isCount = (v: unknown): v is number => finite(v) && Number.isInteger(v) && v >= 0
const isCounts = (v: unknown): v is SplitCounts => isObj(v) && isCount(v.train) && isCount(v.val) && isCount(v.test)
const isImage = (v: unknown): v is string => typeof v === 'string' && v.length === B64_LEN && B64.test(v)

function validClass(c: unknown): c is DatasetClass {
  return isObj(c) && typeof c.label === 'string' && isCounts(c.n) && isImage(c.mean) && Array.isArray(c.samples) && c.samples.length > 0 && c.samples.every(isImage)
}

function validDataset(d: unknown): d is DatasetIntro {
  if (!isObj(d) || typeof d.title !== 'string' || (d.kind !== 'digits' && d.kind !== 'clothing') || !isCounts(d.n)) return false
  if (!finite(d.ink_fraction) || d.ink_fraction < 0 || d.ink_fraction > 1) return false
  if (!Array.isArray(d.classes) || d.classes.length < 2 || !d.classes.every(validClass)) return false
  // the per-class counts must add up to the split sizes, or the bar chart would not describe this dataset
  const n = d.n as SplitCounts
  return (['train', 'val', 'test'] as const).every((s) => (d.classes as DatasetClass[]).reduce((a, c) => a + c.n[s], 0) === n[s])
}

/**
 * Per dataset, what its images look like. The images stay base64 strings here and are decoded only when drawn
 * (`decodeImage`). Anything with another schema, image size or shape is dropped; that dataset then has no
 * introduction figures, and a file that is wholly invalid gives an empty record.
 */
export function parseDatasets(raw: unknown): Record<string, DatasetIntro> {
  if (!isObj(raw) || raw.schema !== DATASETS_SCHEMA || raw.size !== IMAGE_SIZE || !isObj(raw.datasets)) return {}
  const out: Record<string, DatasetIntro> = {}
  for (const [id, d] of Object.entries(raw.datasets)) {
    if (!validDataset(d)) continue
    out[id] = {
      title: d.title,
      kind: d.kind,
      n: { train: d.n.train, val: d.n.val, test: d.n.test },
      ink_fraction: d.ink_fraction,
      classes: d.classes.map((c) => ({ label: c.label, n: { train: c.n.train, val: c.n.val, test: c.n.test }, mean: c.mean, samples: [...c.samples] })),
    }
  }
  return out
}

/** One image: 784 grey levels, row-major. */
export function decodeImage(b64: string): Uint8Array {
  const s = atob(b64)
  const out = new Uint8Array(IMAGE_PIXELS)
  for (let i = 0; i < IMAGE_PIXELS; i++) out[i] = s.charCodeAt(i)
  return out
}
