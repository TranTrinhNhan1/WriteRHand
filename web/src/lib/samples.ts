/**
 * Hand-built stroke templates for the one-click sample digits. They are only drawing input: the models still
 * see them through the normal pipeline (capture, crop, scale, centre), exactly like a stroke drawn with a mouse.
 * Coordinates are fractions of the drawing box.
 */
export interface Pt {
  x: number
  y: number
}

type P = [number, number]

function bez(a: P, b: P, c: P, d: P, n = 14): Pt[] {
  const out: Pt[] = []
  for (let i = 0; i <= n; i++) {
    const t = i / n
    const u = 1 - t
    out.push({
      x: u * u * u * a[0] + 3 * u * u * t * b[0] + 3 * u * t * t * c[0] + t * t * t * d[0],
      y: u * u * u * a[1] + 3 * u * u * t * b[1] + 3 * u * t * t * c[1] + t * t * t * d[1],
    })
  }
  return out
}

function line(a: P, b: P, n = 8): Pt[] {
  return Array.from({ length: n + 1 }, (_, i) => ({ x: a[0] + ((b[0] - a[0]) * i) / n, y: a[1] + ((b[1] - a[1]) * i) / n }))
}

function ellipse(cx: number, cy: number, rx: number, ry: number, from: number, to: number, n = 28): Pt[] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const a = from + ((to - from) * i) / n
    return { x: cx + rx * Math.cos(a), y: cy - ry * Math.sin(a) }
  })
}

const join = (...parts: Pt[][]): Pt[] => parts.flatMap((p, i) => (i === 0 ? p : p.slice(1)))

export const SAMPLE_DIGITS = [2, 3, 4, 7, 9] as const
export type SampleDigit = (typeof SAMPLE_DIGITS)[number]

export const SAMPLE_STROKES: Record<SampleDigit, Pt[][]> = {
  2: [join(bez([0.27, 0.3], [0.3, 0.1], [0.7, 0.08], [0.7, 0.33]), bez([0.7, 0.33], [0.7, 0.52], [0.4, 0.66], [0.27, 0.84]), line([0.27, 0.84], [0.76, 0.84]))],
  3: [
    join(
      bez([0.28, 0.2], [0.45, 0.06], [0.74, 0.14], [0.66, 0.36]),
      bez([0.66, 0.36], [0.62, 0.46], [0.5, 0.48], [0.42, 0.5]),
      bez([0.42, 0.5], [0.74, 0.5], [0.78, 0.8], [0.58, 0.87]),
      bez([0.58, 0.87], [0.46, 0.9], [0.32, 0.86], [0.25, 0.77]),
    ),
  ],
  4: [join(line([0.6, 0.14], [0.24, 0.62], 10), line([0.24, 0.62], [0.78, 0.62], 10)), line([0.62, 0.3], [0.62, 0.88], 10)],
  7: [join(line([0.25, 0.18], [0.76, 0.18], 10), bez([0.76, 0.18], [0.64, 0.4], [0.5, 0.62], [0.42, 0.88]))],
  9: [
    join(ellipse(0.5, 0.33, 0.18, 0.2, 0, Math.PI * 2), bez([0.68, 0.33], [0.68, 0.55], [0.62, 0.74], [0.5, 0.88])),
  ],
}
