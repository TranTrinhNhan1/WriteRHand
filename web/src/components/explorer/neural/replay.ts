/** Pure helpers for the training replay panel (genuine recorded data only; nothing is interpolated). */
import type { ReplayData, ReplaySnapshot, ReplayTensor } from '../../../engine/contract'

/** Measured series values at `epoch`, or null when the epoch is not in the series (e.g. 0 = initial weights). */
export function seriesAt(r: ReplayData, epoch: number): Record<string, number> | null {
  const i = r.epochs.indexOf(epoch)
  if (i < 0) return null
  const out: Record<string, number> = {}
  for (const [k, v] of Object.entries(r.series)) out[k] = v[i]
  return out
}

export type TensorLayout =
  | { kind: 'columns'; side: number; rows: number; cols: number }
  | { kind: 'kernels'; cout: number; cin: number; kh: number; kw: number }
  | { kind: 'matrix'; w: number; h: number }
  | { kind: 'vector'; n: number }

/**
 * How to draw a saved tensor: a [side², k] matrix (first-layer weights) as k images of side × side,
 * a [Cout, Cin, k, k] tensor as kernels, other 2-D tensors as a matrix and 1-D ones as a strip.
 */
export function tensorLayout(t: ReplayTensor, side: number): TensorLayout {
  const s = t.shape
  if (s.length === 2 && s[0] === side * side) return { kind: 'columns', side, rows: s[0], cols: s[1] }
  if (s.length === 4) return { kind: 'kernels', cout: s[0], cin: s[1], kh: s[2], kw: s[3] }
  if (s.length === 2) return { kind: 'matrix', w: s[1], h: s[0] }
  return { kind: 'vector', n: s.reduce((a, b) => a * b, 1) }
}

/** Column `j` of a row-major [rows, cols] tensor. */
export function column(data: ArrayLike<number>, rows: number, cols: number, j: number): Float32Array {
  const out = new Float32Array(rows)
  for (let i = 0; i < rows; i++) out[i] = data[i * cols + j]
  return out
}

/** Largest |value| of a tensor over all snapshots, so colours stay comparable while stepping through time. */
export function tensorMax(snapshots: ReplaySnapshot[], name: string): number {
  let m = 0
  for (const s of snapshots) {
    const t = s.tensors[name]
    if (!t) continue
    for (const v of t.data) m = Math.max(m, Math.abs(v))
  }
  return m || 1
}

/** Index of the largest value; -1 when every value is 0 (e.g. AdaBoost before the first round has no votes). */
export function predicted(row: ArrayLike<number>): number {
  let b = -1
  let bv = 0
  for (let i = 0; i < row.length; i++) if (row[i] > bv) ((b = i), (bv = row[i]))
  return b
}

/** Recorded samples whose top output equals the label at this snapshot. */
export function correctCount(snap: ReplaySnapshot, labels: number[]): number {
  return snap.proba.reduce((n, row, i) => n + (predicted(row) === labels[i] ? 1 : 0), 0)
}

/** Index of the snapshot recorded at `epoch`, or -1. */
export const snapshotAt = (r: ReplayData, epoch: number | null | undefined) => (epoch == null ? -1 : r.snapshots.findIndex((s) => s.epoch === epoch))

/** Pixels of a recorded sample in [0, 1] (files store 0..255). */
export function samplePixels(px: number[]): Float32Array {
  let m = 0
  for (const v of px) m = Math.max(m, v)
  const d = m > 1 ? 255 : 1
  return Float32Array.from(px, (v) => v / d)
}
