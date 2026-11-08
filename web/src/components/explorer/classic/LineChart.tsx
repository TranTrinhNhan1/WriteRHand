import { useEffect, useRef, type KeyboardEvent, type PointerEvent } from 'react'
import { useCssColors, useWidth } from './useCssColors'

interface Props {
  /** (steps + 1) x K values, row-major by step */
  rows: Float64Array
  K: number
  steps: number
  /** current step: lines are drawn strongly up to here */
  cursor: number
  names: string[]
  /** class drawn in the accent colour */
  highlight: number
  /** class leading at the cursor (-1: none) */
  leader: number
  /** draw what comes after the cursor faintly, or not at all */
  future: 'hide' | 'faint'
  /** steps to mark with a tick on the top edge (e.g. where the leader changes) */
  marks?: number[]
  yFormat: (v: number) => string
  /** fixed y domain; defaults to the range of all rows */
  domain?: [number, number]
  height: number
  label: string
  valueText: string
  onSeek?: (step: number) => void
}

const COLORS = ['--ink', '--ink-2', '--ink-3', '--line', '--line-2', '--accent', '--surface'] as const

function niceTicks(lo: number, hi: number, n: number): number[] {
  const span = hi - lo || 1
  const raw = span / n
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return out
}

/**
 * Lines of K series over steps (canvas), with a cursor and class labels at the cursor. Pointer drag and
 * the arrow keys move the cursor (it is a slider for assistive technology).
 */
export function LineChart({ rows, K, steps, cursor, names, highlight, leader, future, marks, yFormat, domain, height, label, valueText, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const width = useWidth(wrap)
  const c = useCssColors(COLORS)
  const geom = useRef({ left: 0, plotW: 1 })

  useEffect(() => {
    const cv = canvas.current
    const ctx = cv?.getContext('2d')
    if (!cv || !ctx || width <= 0) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    cv.width = Math.round(width * dpr)
    cv.height = Math.round(height * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)
    const font = '11px "Geist Mono Variable", ui-monospace, monospace'
    ctx.font = font
    let lo: number
    let hi: number
    if (domain) [lo, hi] = domain
    else {
      lo = Infinity
      hi = -Infinity
      for (let i = 0; i < rows.length; i++) {
        lo = Math.min(lo, rows[i])
        hi = Math.max(hi, rows[i])
      }
    }
    if (!(hi > lo)) {
      hi = lo + 1
    }
    const ticks = niceTicks(lo, hi, 4)
    const left = Math.ceil(Math.max(...ticks.map((v) => ctx.measureText(yFormat(v)).width))) + 8
    const gutter = Math.min(110, Math.ceil(Math.max(...names.map((n) => ctx.measureText(n).width))) + 16)
    const top = 10
    const bottom = height - 18
    const right = width - gutter
    const plotW = Math.max(1, right - left)
    geom.current = { left, plotW }
    const X = (s: number) => left + (steps > 0 ? (s / steps) * plotW : 0)
    const Y = (v: number) => bottom - ((v - lo) / (hi - lo)) * (bottom - top)

    // grid and axis labels
    ctx.lineWidth = 1
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'right'
    for (const v of ticks) {
      const y = Math.round(Y(v)) + 0.5
      ctx.strokeStyle = c['--line']
      ctx.beginPath()
      ctx.moveTo(left, y)
      ctx.lineTo(right, y)
      ctx.stroke()
      ctx.fillStyle = c['--ink-3']
      ctx.fillText(yFormat(v), left - 6, y)
    }
    ctx.textBaseline = 'alphabetic'
    ctx.textAlign = 'left'
    ctx.fillText('0', left, height - 4)
    ctx.textAlign = 'right'
    ctx.fillText(String(steps), right, height - 4)

    // marks on the top edge
    if (marks) {
      ctx.fillStyle = c['--ink-3']
      for (const s of marks) {
        const x = X(s)
        ctx.beginPath()
        ctx.moveTo(x - 3, top - 8)
        ctx.lineTo(x + 3, top - 8)
        ctx.lineTo(x, top - 3)
        ctx.closePath()
        ctx.fill()
      }
    }

    const order = Array.from({ length: K }, (_, k) => k).filter((k) => k !== highlight && k !== leader)
    if (leader >= 0 && leader !== highlight) order.push(leader)
    order.push(highlight)
    const path = (k: number, from: number, to: number) => {
      ctx.beginPath()
      // skip points that fall on the same pixel column
      const stride = Math.max(1, Math.floor((to - from) / (plotW * 2)))
      for (let s = from; s <= to; s += stride) {
        const x = X(s)
        const y = Y(rows[s * K + k])
        if (s === from) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.lineTo(X(to), Y(rows[to * K + k]))
      ctx.stroke()
    }
    ctx.lineJoin = 'round'
    if (future === 'faint' && cursor < steps) {
      ctx.globalAlpha = 0.22
      ctx.strokeStyle = c['--ink-3']
      ctx.lineWidth = 1
      for (const k of order) path(k, cursor, steps)
      ctx.globalAlpha = 1
    }
    if (cursor > 0) {
      for (const k of order) {
        const isHi = k === highlight
        const isLead = k === leader
        ctx.strokeStyle = isHi ? c['--accent'] : isLead ? c['--ink'] : c['--ink-3']
        ctx.globalAlpha = isHi || isLead ? 1 : 0.55
        ctx.lineWidth = isHi ? 2 : isLead ? 1.5 : 1
        path(k, 0, cursor)
      }
      ctx.globalAlpha = 1
    }

    // cursor
    const cx = Math.round(X(cursor)) + 0.5
    ctx.strokeStyle = c['--ink-2']
    ctx.lineWidth = 1
    ctx.setLineDash([3, 3])
    ctx.beginPath()
    ctx.moveTo(cx, top)
    ctx.lineTo(cx, bottom)
    ctx.stroke()
    ctx.setLineDash([])
    for (const k of [leader, highlight]) {
      if (k < 0) continue
      ctx.fillStyle = k === highlight ? c['--accent'] : c['--ink']
      ctx.beginPath()
      ctx.arc(X(cursor), Y(rows[cursor * K + k]), 3, 0, Math.PI * 2)
      ctx.fill()
    }

    // class labels at the cursor values, pushed apart so they never overlap
    const gap = 12
    const labels = Array.from({ length: K }, (_, k) => ({ k, y: Y(rows[cursor * K + k]) })).sort((a, b) => a.y - b.y)
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + gap)
    const n = labels.length
    if (n) labels[n - 1].y = Math.min(labels[n - 1].y, bottom + 4)
    for (let i = n - 2; i >= 0; i--) labels[i].y = Math.min(labels[i].y, labels[i + 1].y - gap)
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    for (const { k, y } of labels) {
      const isHi = k === highlight
      const isLead = k === leader
      ctx.font = `${isHi || isLead ? 600 : 400} 11px "Geist Mono Variable", ui-monospace, monospace`
      ctx.fillStyle = isHi ? c['--accent'] : isLead ? c['--ink'] : c['--ink-3']
      ctx.fillText(names[k], right + 8, y)
    }
  }, [rows, K, steps, cursor, names, highlight, leader, future, marks, yFormat, domain, height, width, c])

  const seekAt = (e: PointerEvent<HTMLDivElement>) => {
    if (!onSeek) return
    const r = e.currentTarget.getBoundingClientRect()
    const { left, plotW } = geom.current
    onSeek(Math.max(0, Math.min(steps, Math.round(((e.clientX - r.left - left) / plotW) * steps))))
  }
  const onKey = (e: KeyboardEvent) => {
    if (!onSeek) return
    const big = Math.max(1, Math.round(steps / 10))
    const d: Record<string, number> = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -big, PageUp: big }
    if (e.key === 'Home') onSeek(0)
    else if (e.key === 'End') onSeek(steps)
    else if (d[e.key]) onSeek(Math.max(0, Math.min(steps, cursor + d[e.key])))
    else return
    e.preventDefault()
  }
  return (
    <div
      ref={wrap}
      className="exc-chart"
      style={{ height }}
      role={onSeek ? 'slider' : 'img'}
      aria-label={label}
      aria-valuemin={onSeek ? 0 : undefined}
      aria-valuemax={onSeek ? steps : undefined}
      aria-valuenow={onSeek ? cursor : undefined}
      aria-valuetext={valueText}
      tabIndex={onSeek ? 0 : undefined}
      onKeyDown={onKey}
      onPointerDown={(e) => {
        if (!onSeek) return
        e.currentTarget.setPointerCapture(e.pointerId)
        seekAt(e)
      }}
      onPointerMove={(e) => {
        if (e.buttons & 1) seekAt(e)
      }}
    >
      <canvas ref={canvas} style={{ width: '100%', height }} aria-hidden="true" />
    </div>
  )
}
