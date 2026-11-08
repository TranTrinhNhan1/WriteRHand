import type { Key } from '../i18n'
import { finite } from './scale'
import { paramValue } from './format'
import type { AugCondition, AugStudy, CnnVsMlp, CrossEval, ModelKind, ResultModel, Results, SplitId, SplitMetrics, SweepPoint, TuningEntry } from './types'

export function splitOf(m: ResultModel, s: SplitId): SplitMetrics | undefined {
  return m.splits?.[s]
}

export interface Confusion {
  trueDigit: number
  predDigit: number
  count: number
  /** Share of all images of the true digit. */
  share: number
}

/** Largest off-diagonal cells of a confusion matrix (rows = true, columns = predicted). */
export function topConfusions(matrix: number[][] | null | undefined, k = 5): Confusion[] {
  if (!matrix) return []
  const out: Confusion[] = []
  matrix.forEach((row, r) => {
    const total = row.reduce((a, b) => a + b, 0)
    row.forEach((count, c) => {
      if (r !== c && count > 0) out.push({ trueDigit: r, predDigit: c, count, share: total ? count / total : 0 })
    })
  })
  return out.sort((a, b) => b.count - a.count || a.trueDigit - b.trueDigit).slice(0, k)
}

export function isValidMatrix(m: unknown): m is number[][] {
  return Array.isArray(m) && m.length > 0 && m.every((r) => Array.isArray(r) && r.length === m.length && r.every(finite))
}

export type SeriesRole = 'train' | 'val' | 'holdout' | 'neutral'

export interface HistorySeries {
  key: string
  labelKey: Key
  role: SeriesRole
  values: number[]
}

export interface HistoryPanel {
  id: string
  titleKey: Key
  yKey: Key
  format: 'pct' | 'num'
  series: HistorySeries[]
}

const LABELS: Record<string, { labelKey: Key; role: SeriesRole; panel: string }> = {
  loss: { labelKey: 'results.curves.series.loss', role: 'train', panel: 'loss' },
  val_loss: { labelKey: 'results.curves.series.val_loss', role: 'val', panel: 'loss' },
  inner_val_loss: { labelKey: 'results.curves.series.inner_val_loss', role: 'holdout', panel: 'loss' },
  train_err_weighted: { labelKey: 'results.curves.series.train_err_weighted', role: 'train', panel: 'loss' },
  train_acc: { labelKey: 'results.curves.series.train_acc', role: 'train', panel: 'acc' },
  val_acc: { labelKey: 'results.curves.series.val_acc', role: 'val', panel: 'acc' },
  alpha: { labelKey: 'results.curves.series.alpha', role: 'neutral', panel: 'alpha' },
}

/** Splits `models[].history` into loss / accuracy / weight panels; tolerates missing or ragged arrays. */
export function historyPanels(id: ModelKind, history: Record<string, number[]> | null | undefined): HistoryPanel[] {
  if (!history) return []
  const boosting = id === 'adaboost'
  const panels: HistoryPanel[] = [
    {
      id: 'loss',
      titleKey: boosting ? 'results.curves.panel.lossRound' : 'results.curves.panel.lossEpoch',
      yKey: boosting ? 'results.curves.panel.yError' : 'results.curves.panel.yLoss',
      format: 'num',
      series: [],
    },
    {
      id: 'acc',
      titleKey: boosting ? 'results.curves.panel.accRound' : 'results.curves.panel.accEpoch',
      yKey: 'results.curves.panel.yAcc',
      format: 'pct',
      series: [],
    },
    { id: 'alpha', titleKey: 'results.curves.panel.alpha', yKey: 'results.curves.panel.yAlpha', format: 'num', series: [] },
  ]
  for (const [key, values] of Object.entries(history)) {
    const meta = LABELS[key]
    if (!meta || !Array.isArray(values) || !values.some(finite)) continue
    panels.find((p) => p.id === meta.panel)!.series.push({ key, labelKey: meta.labelKey, role: meta.role, values })
  }
  return panels.filter((p) => p.series.length > 0)
}

export interface SweepPanel {
  param: string
  kind: 'numeric' | 'categorical'
  log: boolean
  points: { label: string; x: number; train?: number; val?: number; value: unknown }[]
  chosen: { x: number; label: string } | null
}

function sameValue(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

/** One panel per swept hyper-parameter with at least two evaluated values. */
export function sweepPanels(entry: TuningEntry | null | undefined): SweepPanel[] {
  const sweeps = entry?.sweeps
  if (!sweeps) return []
  const out: SweepPanel[] = []
  for (const [param, raw] of Object.entries(sweeps)) {
    const pts = (raw ?? []).filter((p: SweepPoint) => finite(p.val_acc) || finite(p.train_acc))
    if (pts.length < 2) continue
    const numeric = pts.every((p) => finite(p.value))
    const best = entry?.best_params?.[param]
    const ordered = numeric ? [...pts].sort((a, b) => (a.value as number) - (b.value as number)) : pts
    const values = ordered.map((p) => p.value as number)
    const log = numeric && Math.min(...values) > 0 && Math.max(...values) / Math.min(...values) >= 30
    const points = ordered.map((p, i) => ({
      label: paramValue(p.value),
      x: numeric ? (p.value as number) : i,
      train: finite(p.train_acc) ? p.train_acc : undefined,
      val: finite(p.val_acc) ? p.val_acc : undefined,
      value: p.value,
    }))
    const hit = points.find((p) => sameValue(p.value, best))
    out.push({ param, kind: numeric ? 'numeric' : 'categorical', log, points, chosen: hit ? { x: hit.x, label: hit.label } : null })
  }
  return out
}

export interface CurvePanel {
  param: string
  chosen: number | null
  train: number[]
  val: number[]
}

/** Accuracy against epochs / boosting rounds from the tuning run, with the chosen value. */
export function curvePanel(entry: TuningEntry | null | undefined): CurvePanel | null {
  const c = entry?.curve
  if (!c || !Array.isArray(c.val_acc) || c.val_acc.length < 2) return null
  return {
    param: entry?.epoch_curve?.param ?? 'epochs',
    chosen: finite(entry?.epoch_curve?.chosen) ? entry!.epoch_curve!.chosen! : null,
    train: Array.isArray(c.train_acc) ? c.train_acc : [],
    val: c.val_acc,
  }
}

// --------------------------------------------------------------------------- optional sections
const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)

function validAugStudy(s: unknown): s is AugStudy {
  if (!isObj(s) || !isObj(s.conditions)) return false
  const conds = Object.values(s.conditions)
  return conds.length > 0 && conds.every((c) => isObj(c) && finite((c as unknown as AugCondition).val_acc_mean) && Array.isArray((c as unknown as AugCondition).runs))
}

function validCnnVsMlp(v: unknown): v is CnnVsMlp {
  return isObj(v) && Array.isArray(v.rows) && v.rows.length > 0 && v.rows.every((r) => isObj(r) && (r.id === 'mlp' || r.id === 'cnn') && isObj(r.accuracy))
}

function validCross(v: unknown): v is CrossEval {
  return isObj(v) && typeof v.trained_on === 'string' && Array.isArray(v.rows) && v.rows.every((r) => isObj(r) && typeof r.model === 'string' && finite(r.accuracy))
}

/**
 * Keeps the measured numbers as they are and drops optional sections whose shape is not the one
 * the page knows how to draw, so a partial or older results file never breaks a page: a section
 * is shown only when its data exists and is complete.
 */
export function sanitizeResults(raw: unknown): Results {
  const r = (isObj(raw) ? raw : {}) as unknown as Results & Record<string, unknown>
  const out: Results = { ...r, models: Array.isArray(r.models) ? r.models.filter((m) => isObj(m) && typeof m.id === 'string') : [] }
  delete out.augmentation
  delete out.cnn_vs_mlp
  delete out.cross_eval
  if (isObj(r.augmentation)) {
    const aug: NonNullable<Results['augmentation']> = {}
    if (validAugStudy(r.augmentation.mlp)) aug.mlp = r.augmentation.mlp
    if (validAugStudy(r.augmentation.cnn)) aug.cnn = r.augmentation.cnn
    if (aug.mlp || aug.cnn) out.augmentation = aug
  }
  if (validCnnVsMlp(r.cnn_vs_mlp)) out.cnn_vs_mlp = r.cnn_vs_mlp
  if (Array.isArray(r.cross_eval)) {
    const ok = r.cross_eval.filter(validCross)
    if (ok.length) out.cross_eval = ok
  }
  return out
}

export type AugVerdict = 'baseline' | 'helps' | 'unclear' | 'hurts'

export interface AugRow {
  name: string
  ops: string[]
  seeds: number
  valMean: number
  valSd: number
  testMean: number
  testSd: number
  delta: number | null
  deltaSd: number | null
  verdict: AugVerdict
  robust: Record<string, number>
  selected: boolean
}

/**
 * Effect of a condition on validation accuracy, judged on the paired differences with the baseline
 * (same seeds): "helps" when the mean difference is positive and larger than one standard deviation
 * of the differences, "hurts" when negative and larger than one standard deviation in size, otherwise
 * "no clear effect". With fewer than two seeds there is no spread to compare with, so nothing is clear.
 */
export function augVerdict(delta: number | null | undefined, sd: number | null | undefined, seeds: number): Exclude<AugVerdict, 'baseline'> {
  if (!finite(delta) || seeds < 2 || !finite(sd)) return 'unclear'
  if (Math.abs(delta) <= sd || delta === 0) return 'unclear'
  return delta > 0 ? 'helps' : 'hurts'
}

export function augRows(study: AugStudy): AugRow[] {
  return Object.entries(study.conditions).map(([name, c]) => {
    const seeds = c.runs.length
    const isBase = name === 'baseline'
    const delta = finite(c.val_delta_vs_baseline_mean) ? c.val_delta_vs_baseline_mean : null
    const deltaSd = finite(c.val_delta_vs_baseline_sd) ? c.val_delta_vs_baseline_sd : null
    return {
      name,
      ops: c.ops ?? [],
      seeds,
      valMean: c.val_acc_mean,
      valSd: finite(c.val_acc_std) ? c.val_acc_std : 0,
      testMean: c.test_acc_mean,
      testSd: finite(c.test_acc_std) ? c.test_acc_std : 0,
      delta: isBase ? 0 : delta,
      deltaSd: isBase ? 0 : deltaSd,
      verdict: isBase ? 'baseline' : augVerdict(delta, deltaSd, seeds),
      robust: c.robust_test_acc_mean ?? {},
      selected: study.selected_by_validation === name,
    }
  })
}

/** Stress perturbations of the robustness benchmark, in the order the study lists them. */
export function augStressKeys(study: AugStudy): string[] {
  const keys = new Set<string>(Object.keys(study.stress ?? {}))
  for (const c of Object.values(study.conditions)) for (const k of Object.keys(c.robust_test_acc_mean ?? {})) keys.add(k)
  return [...keys].filter((k) => Object.values(study.conditions).some((c) => finite(c.robust_test_acc_mean?.[k])))
}

export interface DatasetCell {
  accuracy: number | null
  f1: number | null
}

export interface DatasetMatrix {
  datasets: string[]
  models: ModelKind[]
  cells: Record<string, Partial<Record<ModelKind, DatasetCell>>>
}

/** Test accuracy and macro-F1 of every model on every dataset (rows: models, columns: datasets). */
export function datasetMatrix(results: Record<string, Results>, datasets: string[], order: readonly ModelKind[]): DatasetMatrix {
  const cells: DatasetMatrix['cells'] = {}
  const present = new Set<ModelKind>()
  for (const ds of datasets) {
    cells[ds] = {}
    for (const m of results[ds]?.models ?? []) {
      const test = m.splits?.test
      const accuracy = finite(test?.accuracy) ? test!.accuracy! : null
      const f1 = finite(test?.f1_macro) ? test!.f1_macro! : null
      if (accuracy == null && f1 == null) continue
      cells[ds][m.id] = { accuracy, f1 }
      present.add(m.id)
    }
  }
  return { datasets, models: order.filter((m) => present.has(m)), cells }
}

export interface DatasetSpread {
  dataset: string
  best: { model: ModelKind; accuracy: number } | null
  worst: { model: ModelKind; accuracy: number } | null
  /** best minus worst test accuracy, in percentage points */
  gapPoints: number | null
}

/** Neutral observations computed from the numbers only: best and lowest test accuracy per dataset. */
export function datasetSpreads(m: DatasetMatrix): DatasetSpread[] {
  return m.datasets.map((ds) => {
    const entries = m.models.flatMap((id) => {
      const a = m.cells[ds]?.[id]?.accuracy
      return a == null ? [] : [{ model: id, accuracy: a }]
    })
    if (!entries.length) return { dataset: ds, best: null, worst: null, gapPoints: null }
    const best = entries.reduce((a, b) => (b.accuracy > a.accuracy ? b : a))
    const worst = entries.reduce((a, b) => (b.accuracy < a.accuracy ? b : a))
    return { dataset: ds, best, worst, gapPoints: entries.length > 1 ? (best.accuracy - worst.accuracy) * 100 : null }
  })
}
