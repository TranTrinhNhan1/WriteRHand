import { useEffect, useRef, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react'

export type HeatMode = 'diverging' | 'gray' | 'active'

export interface HeatMapProps {
  /** Row-major cell values (length width * height). */
  values: ArrayLike<number>
  width: number
  height: number
  mode: HeatMode
  /** Largest magnitude mapped to full colour. Defaults to the largest |value|. */
  max?: number
  /** Accessible name. */
  label: string
  /** Cells outlined as "selected" (e.g. the pixel whose contribution is shown) and "marked" (e.g. pixels a tree looked at). */
  selected?: number | null
  marks?: number[]
  /** A rectangle in cell units, e.g. the receptive field of an output location or a sliding kernel window. */
  box?: { x0: number; y0: number; x1: number; y1: number } | null
  onHover?: (cell: number | null) => void
  onSelect?: (cell: number) => void
  className?: string
  style?: CSSProperties
}

// Heat maps sit on the dark "instrument screen" in both themes, so the palette is fixed.
const SCREEN = [7, 9, 8]
const POS = [198, 241, 53]
const NEG = [255, 122, 89]
const WHITE = [241, 245, 236]
const mix = (base: number[], col: number[], a: number) => {
  const t = Math.max(0, Math.min(1, a))
  return [base[0] + (col[0] - base[0]) * t, base[1] + (col[1] - base[1]) * t, base[2] + (col[2] - base[2]) * t]
}

/**
 * A small matrix of cells (one canvas pixel per cell, scaled by CSS, no smoothing) with optional
 * hover / click selection, outlines and a rectangle overlay. Keyboard: arrow keys move the selection.
 */
export function HeatMap({ values, width, height, mode, max, label, selected, marks, box, onHover, onSelect, className, style }: HeatMapProps) {
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
      const rgb = mode === 'diverging' ? mix(SCREEN, v >= 0 ? POS : NEG, Math.abs(v) / top) : mode === 'gray' ? mix(SCREEN, WHITE, v / top) : mix(SCREEN, POS, v / top)
      img.data.set([rgb[0], rgb[1], rgb[2], 255], 4 * i)
    }
    ctx.putImageData(img, 0, 0)
  }, [values, width, height, mode, max])

  const cellAt = (e: PointerEvent<HTMLElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const x = Math.min(width - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * width)))
    const y = Math.min(height - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * height)))
    return y * width + x
  }
  const onKey = (e: KeyboardEvent) => {
    if (!onSelect) return
    const cur = selected ?? 0
    const x = cur % width
    const y = Math.floor(cur / width)
    const d: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }
    const m = d[e.key]
    if (!m) return
    e.preventDefault()
    onSelect(Math.min(height - 1, Math.max(0, y + m[1])) * width + Math.min(width - 1, Math.max(0, x + m[0])))
  }
  const pct = (v: number, n: number) => `${(100 * v) / n}%`
  const rect = (c: number) => ({ left: pct(c % width, width), top: pct(Math.floor(c / width), height), width: pct(1, width), height: pct(1, height) })
  return (
    <div
      className={`ex-heat ${className ?? ''}`}
      style={{ aspectRatio: `${width} / ${height}`, ...style }}
      role={onSelect ? 'application' : 'img'}
      aria-label={label}
      tabIndex={onSelect ? 0 : undefined}
      onKeyDown={onKey}
      onPointerMove={onHover ? (e) => onHover(cellAt(e)) : undefined}
      onPointerLeave={onHover ? () => onHover(null) : undefined}
      onClick={onSelect ? (e) => onSelect(cellAt(e as unknown as PointerEvent<HTMLElement>)) : undefined}
    >
      <canvas ref={ref} width={width} height={height} />
      {marks?.map((c) => (
        <span key={c} className="ex-heat__mark" style={rect(c)} />
      ))}
      {box && (
        <span
          className="ex-heat__box"
          style={{ left: pct(box.x0, width), top: pct(box.y0, height), width: pct(box.x1 - box.x0, width), height: pct(box.y1 - box.y0, height) }}
        />
      )}
      {selected != null && <span className="ex-heat__sel" style={rect(selected)} />}
    </div>
  )
}
