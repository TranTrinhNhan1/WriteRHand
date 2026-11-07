import type { Key } from '../i18n'
import { finite } from './scale'
import { paramValue } from './format'
import type { ModelId, ResultModel, SplitId, SplitMetrics, SweepPoint, TuningEntry } from './types'

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
export function historyPanels(id: ModelId, history: Record<string, number[]> | null | undefined): HistoryPanel[] {
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
