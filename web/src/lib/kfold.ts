import { finite } from './scale'
import { MODEL_ORDER, type ModelKind } from '../engine/contract'

/** `public/data/kfold.json`, written by `python -m src site-extra` (src/experiments/extras_report.py:kfold_site_json). */
export const KFOLD_SCHEMA = 'writerhand.kfold/1'
/** The k-fold experiment (src/experiments/kfold.py) runs on MNIST only; the file does not name the dataset. */
export const KFOLD_DATASET = 'mnist'

export interface KFoldFold {
  fold: number
  n_train: number
  n_val: number
  val_accuracy: number
  val_f1_macro?: number | null
  test_accuracy: number
  test_f1_macro?: number | null
  train_time_sec?: number | null
}

export interface MeanSd {
  mean: number
  std: number
}

export interface KFoldModel {
  id: ModelKind
  name: string
  params: Record<string, unknown>
  folds: KFoldFold[]
  val_accuracy: MeanSd
  test_accuracy: MeanSd
}

export interface KFold {
  k: number
  seed: number | null
  n_pool: number | null
  n_test: number | null
  note: string
  /** in MODEL_ORDER */
  models: KFoldModel[]
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)
const isAcc = (v: unknown): v is number => finite(v) && v >= 0 && v <= 1
const isMeanSd = (v: unknown): v is MeanSd => isObj(v) && isAcc(v.mean) && finite(v.std) && v.std >= 0

function validFold(f: unknown): f is KFoldFold {
  return isObj(f) && Number.isInteger(f.fold) && finite(f.n_train) && finite(f.n_val) && isAcc(f.val_accuracy) && isAcc(f.test_accuracy)
}

function validModel(m: unknown, k: number): m is KFoldModel {
  if (!isObj(m) || typeof m.id !== 'string' || !(MODEL_ORDER as string[]).includes(m.id)) return false
  return Array.isArray(m.folds) && m.folds.length === k && m.folds.every(validFold) && isMeanSd(m.val_accuracy) && isMeanSd(m.test_accuracy)
}

/**
 * The per-fold measurements of the k-fold cross-validation. Anything with another schema or shape gives null
 * (the section is then not shown); a model whose entry is malformed, or that appears twice, is dropped.
 */
export function parseKFold(raw: unknown): KFold | null {
  if (!isObj(raw) || raw.schema !== KFOLD_SCHEMA || !Number.isInteger(raw.k) || (raw.k as number) < 2 || !Array.isArray(raw.models)) return null
  const k = raw.k as number
  const seen = new Set<string>()
  const models: KFoldModel[] = []
  for (const m of raw.models) {
    if (!validModel(m, k) || seen.has(m.id)) continue
    seen.add(m.id)
    models.push({
      id: m.id,
      name: typeof m.name === 'string' ? m.name : m.id,
      params: isObj(m.params) ? m.params : {},
      folds: [...m.folds].sort((a, b) => a.fold - b.fold),
      val_accuracy: m.val_accuracy,
      test_accuracy: m.test_accuracy,
    })
  }
  if (!models.length) return null
  models.sort((a, b) => MODEL_ORDER.indexOf(a.id) - MODEL_ORDER.indexOf(b.id))
  const opt = (v: unknown) => (finite(v) ? v : null)
  return { k, seed: opt(raw.seed), n_pool: opt(raw.n_pool), n_test: opt(raw.n_test), note: typeof raw.note === 'string' ? raw.note : '', models }
}
