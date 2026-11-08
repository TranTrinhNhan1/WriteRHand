import { useEffect, useRef } from 'react'

export type HeatMode = 'diverging' | 'gray' | 'active'

interface Props {
  values: ArrayLike<number>
  width: number
  height: number
  mode: HeatMode
  /** Scale: the largest magnitude that maps to full colour. Defaults to the largest value in `values`. */
  max?: number
  /** Cells outlined in the accent colour, with a number each (AdaBoost: the pixels a tree looked at). */
  marks?: { index: number; label?: string }[]
  label: string
  className?: string
}

// The maps sit on the dark "instrument screen", whatever the theme, so the palette is fixed.
const SCREEN = [7, 9, 8]
const POS = [198, 241, 53] // lime: supports
const NEG = [255, 122, 89] // coral: speaks against
const WHITE = [241, 245, 236]

function mix(base: number[], col: number[], a: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, a))
  return [base[0] + (col[0] - base[0]) * t, base[1] + (col[1] - base[1]) * t, base[2] + (col[2] - base[2]) * t]
}

/** A small matrix of cells, drawn without smoothing. One canvas pixel per cell, scaled up by CSS. */
export function HeatMap({ values, width, height, mode, max, marks, label, className }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    const n = width * height
    let top = max ?? 0
    if (max == null) for (let i = 0; i < n; i++) top = Math.max(top, Math.abs(values[i] ?? 0))
    if (top <= 0) top = 1
    const img = ctx.createImageData(width, height)
    for (let i = 0; i < n; i++) {
      const v = values[i] ?? 0
      let rgb: [number, number, number]
      if (mode === 'diverging') rgb = mix(SCREEN, v >= 0 ? POS : NEG, Math.abs(v) / top)
      else if (mode === 'gray') rgb = mix(SCREEN, WHITE, v / top)
      else rgb = mix(SCREEN, POS, v / top)
      img.data[4 * i] = rgb[0]
      img.data[4 * i + 1] = rgb[1]
      img.data[4 * i + 2] = rgb[2]
      img.data[4 * i + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [values, width, height, mode, max])

  const cell = 100 / width
  return (
    <div className={`heat${className ? ` ${className}` : ''}`} style={{ aspectRatio: `${width} / ${height}` }}>
      <canvas ref={ref} width={width} height={height} role="img" aria-label={label} />
      {marks?.map((m) => (
        <i
          key={m.index}
          className="heat__mark"
          aria-hidden="true"
          style={{ left: `${(m.index % width) * cell}%`, top: `${Math.floor(m.index / width) * (100 / height)}%`, width: `${cell}%`, height: `${100 / height}%` }}
        >
          {m.label != null && <b>{m.label}</b>}
        </i>
      ))}
    </div>
  )
}
