/** Small helpers for hand-built SVG charts. */

export function niceStep(range: number, target: number): number {
  const raw = range / Math.max(1, target)
  const mag = 10 ** Math.floor(Math.log10(raw))
  const f = raw / mag
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10
  return nice * mag
}

/** Round tick values covering [min, max]; the returned range extends to the outer ticks. */
export function linearTicks(min: number, max: number, target = 5): number[] {
  if (!(max > min)) return [min]
  const step = niceStep(max - min, target)
  const start = Math.floor(min / step + 1e-9) * step
  const out: number[] = []
  for (let v = start; ; v += step) {
    out.push(Number(v.toPrecision(12)))
    if (v >= max - step * 1e-9) break
  }
  return out
}

/** Powers of ten covering [min, max] (both > 0). */
export function logTicks(min: number, max: number): number[] {
  const lo = Math.floor(Math.log10(min) + 1e-9)
  const hi = Math.ceil(Math.log10(max) - 1e-9)
  const out: number[] = []
  for (let e = lo; e <= hi; e++) out.push(10 ** e)
  return out.length > 1 ? out : [10 ** lo, 10 ** (lo + 1)]
}

export interface Scale {
  (v: number): number
  domain: [number, number]
}

export function makeScale(kind: 'linear' | 'log', domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain
  const [r0, r1] = range
  const f =
    kind === 'log'
      ? (v: number) => r0 + ((Math.log10(Math.max(v, 1e-300)) - Math.log10(d0)) / (Math.log10(d1) - Math.log10(d0))) * (r1 - r0)
      : (v: number) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0)
  const s = f as Scale
  s.domain = domain
  return s
}

/** Tick label for the axis: 0.001 -> "0.001", 1e-5 -> "1e−5", 1000 -> "1000". */
export function tickLabel(v: number): string {
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e5 || a < 1e-3) return v.toExponential(0).replace('e-', 'e−').replace('e+', 'e')
  return String(Number(v.toPrecision(6)))
}

export function finite(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

/** Colour of a row-normalised confusion cell: opacity grows with the share (gamma < 1 lifts small values). */
export function cellOpacity(share: number, gamma = 0.7): number {
  return Math.pow(Math.min(1, Math.max(0, share)), gamma)
}
