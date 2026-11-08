/**
 * Pure helpers for choosing a dataset and a model from the manifest, the loading policy and the
 * consensus of the models' answers. Nothing here knows how many datasets or models exist: it reads
 * the manifest.
 */
import { MODEL_ORDER, augId, isAugmented, kindOf, modelIdsOf, type BundleManifest, type DatasetEntry, type ModelEntry, type ModelId, type ModelKind } from '../engine/contract'

/** Models up to this download size load as soon as someone reaches for the pen; larger ones wait for a click. */
export const AUTO_LOAD_MAX_BYTES = 6 * 1024 * 1024

/** The explorer opens on the first of these that the dataset offers. */
const EXPLORER_PREFERENCE: ModelKind[] = ['cnn', 'mlp', 'softmax', 'naive_bayes', 'adaboost']

export interface ModelChoice {
  id: ModelId
  /** architecture: picks the view, the texts and the pairing */
  kind: ModelKind
  /** the variant trained on augmented images */
  augmented: boolean
  entry: ModelEntry
  /** false when the export check failed: the model is listed but cannot be picked */
  selectable: boolean
  /** download size, in bytes, of everything the browser fetches for this model */
  bytes: number
  autoLoad: boolean
}

export function downloadBytes(entry: ModelEntry): number {
  return (entry.size_bytes ?? 0) + (entry.trace.params?.nbytes ?? 0)
}

/** Every model of the dataset. Augmented variants never load by themselves: each waits for a click. */
export function modelChoices(ds: DatasetEntry): ModelChoice[] {
  return modelIdsOf(ds).map((id) => {
    const entry = ds.models[id]!
    const bytes = downloadBytes(entry)
    const selectable = entry.validation?.ok === true
    const augmented = isAugmented(entry)
    return { id, kind: kindOf(entry), augmented, entry, selectable, bytes, autoLoad: selectable && !augmented && bytes <= AUTO_LOAD_MAX_BYTES }
  })
}

export interface ModelPair {
  kind: ModelKind
  plain: ModelChoice | null
  augmented: ModelChoice | null
}

/** One row per architecture: the plain model and its augmented variant (null when the dataset has none). */
export function modelPairs(choices: readonly ModelChoice[]): ModelPair[] {
  return MODEL_ORDER.flatMap((kind) => {
    const plain = choices.find((c) => c.kind === kind && !c.augmented) ?? null
    const augmented = choices.find((c) => c.kind === kind && c.augmented) ?? null
    return plain || augmented ? [{ kind, plain, augmented }] : []
  })
}

/** The id of `kind` in the wanted variant, when the dataset has it. */
export function variantId(ds: DatasetEntry, kind: ModelKind, augmented: boolean): ModelId | null {
  const id = augmented ? augId(kind) : kind
  return ds.models[id] ? id : null
}

export function datasetIdsOf(m: BundleManifest): string[] {
  const ids = Object.keys(m.datasets)
  // default first, the rest in manifest order
  return [m.default_dataset, ...ids.filter((id) => id !== m.default_dataset)].filter((id) => m.datasets[id])
}

export function resolveDataset(m: BundleManifest, wanted: string | null | undefined): string {
  return wanted && m.datasets[wanted] ? wanted : m.default_dataset
}

/** The wanted model when this dataset offers it and it passed the export check, otherwise the preferred available one. */
export function resolveModel(ds: DatasetEntry, wanted: ModelId | null | undefined): ModelId | null {
  const ok = modelChoices(ds).filter((c) => c.selectable)
  if (wanted && ok.some((c) => c.id === wanted)) return wanted
  for (const id of EXPLORER_PREFERENCE) if (ok.some((c) => c.id === id)) return id
  return null
}

export const isDigitDataset = (ds: DatasetEntry) => ds.kind === 'digits'

/** True when every class name is a single character (digits): such labels fit under narrow bars. */
export const shortClassNames = (classes: readonly string[]) => classes.every((c) => c.length <= 2)

export interface Answer {
  model: ModelId
  top: number
  /** value of the top class (a probability, or a vote share for AdaBoost) */
  score: number
}

export interface Verdict {
  cls: number
  votes: number
  of: number
  tied: boolean
}

/**
 * Majority vote of the models' top classes. A tie on the number of votes is broken by the mean
 * top-class score of the tied classes (and flagged, since those scores are not all calibrated).
 */
export function consensusOf(answers: readonly Answer[]): Verdict | null {
  if (answers.length === 0) return null
  const tally = new Map<number, { n: number; s: number }>()
  for (const a of answers) {
    const cur = tally.get(a.top) ?? { n: 0, s: 0 }
    tally.set(a.top, { n: cur.n + 1, s: cur.s + a.score })
  }
  const rows = [...tally.entries()].map(([cls, v]) => ({ cls, n: v.n, mean: v.s / v.n }))
  rows.sort((a, b) => b.n - a.n || b.mean - a.mean || a.cls - b.cls)
  const best = rows[0]
  return { cls: best.cls, votes: best.n, of: answers.length, tied: rows.length > 1 && rows[1].n === best.n }
}

export { MODEL_ORDER }
