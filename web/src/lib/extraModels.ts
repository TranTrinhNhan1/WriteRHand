import { finite } from './scale'
import { MODEL_ORDER, type ModelKind } from '../engine/contract'
import { EXTRA_ORDER, type ExtraModelId, type ResultModel, type TuningEntry } from './types'

/** `public/data/extra_models.json`, written by `python -m src site-extra` (src/experiments/report.py). */
export const EXTRA_SCHEMA = 'writerhand.extra/1'

export interface PairSide {
  train_acc?: number | null
  val_acc?: number | null
  test_acc: number
  test_f1_macro?: number | null
  fit_time_sec?: number | null
  latency_ms?: number | null
  /** test accuracy with every image rotated by a random angle within ±a degrees, keyed by a */
  test_rotation_acc: Record<string, number>
  test_stress_acc?: Record<string, number> | null
}

/** Plain model and its variant trained on augmented images (results/augmentation_pairs.json). */
export interface Pairs {
  rotation_angles: number[]
  recipe: { ops: string[]; copies?: number | null; ranges: Record<string, number | number[] | null> } | null
  models: Partial<Record<ModelKind, { plain: PairSide; augmented: PairSide }>>
}

export interface ExtraResults {
  models: ResultModel[]
  tuning: Partial<Record<ExtraModelId, TuningEntry>>
  pairs?: Pairs
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)
const isExtraId = (v: unknown): v is ExtraModelId => typeof v === 'string' && (EXTRA_ORDER as string[]).includes(v)

function validModel(m: unknown): m is ResultModel {
  if (!isObj(m) || !isExtraId(m.id) || !isObj(m.splits)) return false
  const test = m.splits.test
  if (!isObj(test) || !finite(test.accuracy)) return false
  if (m.id === 'stacking' && !(isObj(m.stacking) && m.stacking.fitted_on === 'val' && Array.isArray(m.stacking.bases))) return false
  return true
}

function validSide(v: unknown): v is PairSide {
  return isObj(v) && finite(v.test_acc) && isObj(v.test_rotation_acc) && Object.values(v.test_rotation_acc).every(finite)
}

function parsePairs(v: unknown): Pairs | undefined {
  if (!isObj(v) || !isObj(v.models) || !Array.isArray(v.rotation_angles) || !v.rotation_angles.every(finite)) return undefined
  const models: Pairs['models'] = {}
  for (const k of MODEL_ORDER) {
    const p = v.models[k]
    if (isObj(p) && validSide(p.plain) && validSide(p.augmented)) models[k] = { plain: p.plain, augmented: p.augmented }
  }
  if (!Object.keys(models).length) return undefined
  const recipe = isObj(v.recipe) && Array.isArray(v.recipe.ops) && isObj(v.recipe.ranges) ? (v.recipe as Pairs['recipe']) : null
  return { rotation_angles: v.rotation_angles as number[], recipe, models }
}

/**
 * Per dataset, the measured models that are not in the browser bundle. Anything with another schema or shape
 * is dropped, never half-drawn; such a model then simply does not appear.
 */
export function parseExtraModels(raw: unknown): Record<string, ExtraResults> {
  if (!isObj(raw) || raw.schema !== EXTRA_SCHEMA || !isObj(raw.datasets)) return {}
  const out: Record<string, ExtraResults> = {}
  for (const [ds, v] of Object.entries(raw.datasets)) {
    if (!isObj(v)) continue
    const models = (Array.isArray(v.models) ? v.models : []).filter(validModel).map((m) => ({ ...m, in_browser: m.in_browser === true }))
    const pairs = parsePairs(v.pairs)
    if (!models.length && !pairs) continue
    const tuning: ExtraResults['tuning'] = {}
    if (isObj(v.tuning)) for (const [k, t] of Object.entries(v.tuning)) if (isExtraId(k) && isObj(t) && models.some((m) => m.id === k)) tuning[k] = t as TuningEntry
    out[ds] = { models: models.sort((a, b) => EXTRA_ORDER.indexOf(a.id as ExtraModelId) - EXTRA_ORDER.indexOf(b.id as ExtraModelId)), tuning, ...(pairs ? { pairs } : {}) }
  }
  return out
}
