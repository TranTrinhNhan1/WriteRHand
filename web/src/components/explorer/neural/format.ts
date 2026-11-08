/** Number formatting for the explorer readouts (real minus sign, fixed digits, compact exponent). */
const MINUS = '−'

/** Signed fixed-point number, e.g. "−0.4240" / "+1.2000" (sign only when `signed`). */
export function fx(v: number, digits = 4, signed = false): string {
  if (!Number.isFinite(v)) return String(v)
  const a = Math.abs(v)
  let s: string
  if (a !== 0 && (a >= 1e5 || a < 10 ** -digits)) s = a.toExponential(2).replace('e-', `e${MINUS}`).replace('e+', 'e')
  else s = a.toFixed(digits)
  if (v < 0 && Number(s) !== 0) return MINUS + s
  if (v < 0 && /e/.test(s)) return MINUS + s
  return signed ? `+${s}` : s
}

/** Absolute difference between a rebuilt value and the runtime value: always an exponent. */
export function diff(v: number): string {
  if (v === 0) return '0'
  return v.toExponential(1).replace('e-', `e${MINUS}`).replace('e+', 'e')
}

/** Shape like "8×26×26". */
export const shape = (s: readonly (number | string)[]) => s.join('×')

/** Integer with thin-space grouping from 5 digits. */
export function int(v: number): string {
  const s = String(Math.abs(Math.round(v)))
  const g = s.length >= 5 ? s.replace(/\B(?=(\d{3})+(?!\d))/g, ' ') : s
  return v < 0 ? MINUS + g : g
}
