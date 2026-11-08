import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { useI18n } from '../../../i18n'
import { usePrefersReducedMotion } from '../../../engine/react'
import { fx } from './format'

export interface GraphColumn {
  key: string
  /** short header, e.g. "h1 · 512" */
  label: string
  /** every value of the layer for the current input (index = neuron) */
  values: ArrayLike<number>
  /** neuron indices drawn (ascending); a subset when the layer is too large */
  shown: number[]
  /** colour rule: standardised inputs and logits are signed, ReLU outputs are >= 0 */
  signed: boolean
  /** names written next to the neurons (output layer) */
  names?: string[]
  /** playback: values of this column have been computed / the weighted sums into it are running */
  filled: boolean
  wired: boolean
}

export interface GraphEdge {
  from: number
  to: number
  /** a_i * W_ij for the current input */
  v: number
}

export interface GraphSel {
  col: number
  neuron: number
  /** source neuron of a selected connection (in column col - 1) */
  from: number | null
}

interface Props {
  columns: GraphColumn[]
  /** edges[c] = connections drawn into column c (edges[0] is empty) */
  edges: GraphEdge[][]
  sel: GraphSel | null
  onSelect: (s: GraphSel) => void
  label: string
}

interface Colors {
  pos: string
  neg: string
  ink: string
  ink3: string
  line: string
  surface: string
  sunken: string
}

const readColors = (el: Element): Colors => {
  const cs = getComputedStyle(el)
  const v = (n: string) => cs.getPropertyValue(n).trim()
  return { pos: v('--accent'), neg: v('--danger'), ink: v('--ink'), ink3: v('--ink-3'), line: v('--line-2'), surface: v('--surface'), sunken: v('--sunken') }
}

const DUR = 420

/**
 * Columns of neurons with the strongest weighted connections between them (Canvas 2D; there can be
 * thousands of lines). Only real neurons and real a·w values are drawn; the caller decides the subset.
 * Mouse / touch: hover or tap a neuron or a line. Keyboard: arrows move the selection.
 */
export function NetworkGraph({ columns, edges, sel, onSelect, label }: Props) {
  const { t } = useI18n()
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [width, setWidth] = useState(0)
  const [hover, setHover] = useState<GraphSel | null>(null)
  const [theme, setTheme] = useState(0)
  const reduced = usePrefersReducedMotion()
  const anim = useRef({ fill: [] as number[], wire: [] as number[], raf: 0, last: 0 })

  useLayoutEffect(() => {
    const el = wrap.current
    if (!el) return
    setWidth(el.getBoundingClientRect().width)
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(el)
    const mo = new MutationObserver(() => setTheme((n) => n + 1))
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => {
      ro.disconnect()
      mo.disconnect()
    }
  }, [])

  const C = columns.length
  const rows = Math.max(...columns.map((c) => c.shown.length))
  const gap = width < 520 ? 9 : 11
  const H = rows * gap + 16
  const hasNames = columns.some((c) => c.names)
  const mL = 14
  const mR = hasNames ? (width < 520 ? 64 : 92) : 14
  const colX = (c: number) => mL + 6 + (C === 1 ? 0 : (c * (width - mL - mR - 12)) / (C - 1))
  const rowY = (c: number, k: number) => 8 + ((k + 0.5) * (H - 16)) / columns[c].shown.length
  const radius = Math.max(2.5, Math.min(5, gap * 0.4))

  // geometry used by drawing and hit testing
  const posOf = (c: number, neuron: number) => {
    const k = columns[c].shown.indexOf(neuron)
    return k < 0 ? null : { x: colX(c), y: rowY(c, k) }
  }

  const draw = useRef<() => void>(() => undefined)
  draw.current = () => {
    const cv = canvas.current
    if (!cv || width === 0) return
    const dpr = window.devicePixelRatio || 1
    if (cv.width !== Math.round(width * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(width * dpr)
      cv.height = Math.round(H * dpr)
    }
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, H)
    const col = readColors(cv)
    const { fill, wire } = anim.current
    // connections
    for (let c = 1; c < C; c++) {
      const p = wire[c] ?? 0
      if (p <= 0) continue
      const list = edges[c] ?? []
      let max = 0
      for (const e of list) max = Math.max(max, Math.abs(e.v))
      if (max === 0) max = 1
      const focus = sel && (sel.col === c || sel.col === c - 1)
      for (const pass of [0, 1]) {
        for (const e of list) {
          const touches = !!sel && ((sel.col === c && sel.neuron === e.to) || (sel.col === c - 1 && sel.neuron === e.from))
          if ((pass === 1) !== touches) continue
          const a = posOf(c - 1, e.from)
          const b = posOf(c, e.to)
          if (!a || !b) continue
          const m = Math.abs(e.v) / max
          const isSel = !!sel && sel.col === c && sel.neuron === e.to && sel.from === e.from
          ctx.globalAlpha = (0.1 + 0.75 * m) * (focus && !touches ? 0.28 : 1)
          ctx.strokeStyle = isSel ? col.ink : e.v >= 0 ? col.pos : col.neg
          ctx.lineWidth = (0.5 + 2.6 * m) * (isSel ? 1.6 : 1)
          ctx.setLineDash(e.v >= 0 ? [] : [3, 2.5])
          ctx.beginPath()
          ctx.moveTo(a.x + radius, a.y)
          ctx.lineTo(a.x + radius + (b.x - radius - a.x - radius) * p, a.y + (b.y - a.y) * p)
          ctx.stroke()
        }
      }
    }
    ctx.setLineDash([])
    ctx.globalAlpha = 1
    // neurons
    ctx.font = `500 ${width < 520 ? 10 : 11}px ${getComputedStyle(cv).getPropertyValue('--sans') || 'sans-serif'}`
    ctx.textBaseline = 'middle'
    for (let c = 0; c < C; c++) {
      const column = columns[c]
      let max = 0
      for (const i of column.shown) max = Math.max(max, Math.abs(column.values[i]))
      if (max === 0) max = 1
      const f = fill[c] ?? 0
      column.shown.forEach((i, k) => {
        const x = colX(c)
        const y = rowY(c, k)
        const v = column.values[i]
        ctx.globalAlpha = 1
        ctx.fillStyle = col.sunken
        ctx.beginPath()
        ctx.arc(x, y, radius, 0, Math.PI * 2)
        ctx.fill()
        if (f > 0) {
          ctx.globalAlpha = f * Math.min(1, 0.12 + Math.abs(v) / max)
          ctx.fillStyle = v >= 0 ? col.pos : col.neg
          ctx.fill()
        }
        ctx.globalAlpha = 1
        ctx.lineWidth = 1
        ctx.strokeStyle = col.line
        ctx.stroke()
        const isSel = sel && sel.col === c && sel.neuron === i
        const isFrom = sel && sel.from === i && sel.col === c + 1
        const isHover = hover && hover.col === c && hover.neuron === i && hover.from == null
        if (isSel || isFrom || isHover) {
          ctx.lineWidth = isSel ? 2 : 1.5
          ctx.strokeStyle = isSel || isFrom ? col.ink : col.ink3
          ctx.beginPath()
          ctx.arc(x, y, radius + 3, 0, Math.PI * 2)
          ctx.stroke()
        }
        if (column.names) {
          ctx.fillStyle = sel && sel.col === c && sel.neuron === i ? col.ink : col.ink3
          const name = column.names[i] ?? String(i)
          ctx.fillText(name.length > 11 ? `${name.slice(0, 10)}…` : name, x + radius + 6, y)
        }
      })
    }
    ctx.globalAlpha = 1
  }

  // playback targets -> short tweens (instant with reduced motion)
  useEffect(() => {
    const a = anim.current
    const target = (c: number, kind: 'fill' | 'wire') => (kind === 'fill' ? (columns[c].filled ? 1 : 0) : columns[c].wired ? 1 : 0)
    if (reduced) {
      a.fill = columns.map((_, c) => target(c, 'fill'))
      a.wire = columns.map((_, c) => target(c, 'wire'))
      draw.current()
      return
    }
    cancelAnimationFrame(a.raf)
    a.last = performance.now()
    const tick = (now: number) => {
      const dt = (now - a.last) / DUR
      a.last = now
      let moving = false
      for (let c = 0; c < C; c++) {
        for (const kind of ['fill', 'wire'] as const) {
          const arr = a[kind]
          const cur = arr[c] ?? 0
          const goal = target(c, kind)
          const next = goal > cur ? Math.min(goal, cur + dt) : Math.max(goal, cur - dt * 3)
          arr[c] = next
          if (next !== goal) moving = true
        }
      }
      draw.current()
      if (moving) a.raf = requestAnimationFrame(tick)
    }
    a.raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(a.raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [columns.map((c) => `${c.filled ? 1 : 0}${c.wired ? 1 : 0}`).join(''), reduced, C])

  useEffect(() => {
    draw.current()
  })
  useEffect(() => draw.current(), [theme])

  const hit = (e: PointerEvent<HTMLElement>): GraphSel | null => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - r.left
    const py = e.clientY - r.top
    let bestC = -1
    let bestDx = Infinity
    for (let c = 0; c < C; c++) {
      const dx = Math.abs(colX(c) - px)
      if (dx < bestDx) {
        bestDx = dx
        bestC = c
      }
    }
    if (bestC >= 0 && bestDx <= Math.max(14, radius * 3)) {
      const shown = columns[bestC].shown
      let bk = 0
      for (let k = 1; k < shown.length; k++) if (Math.abs(rowY(bestC, k) - py) < Math.abs(rowY(bestC, bk) - py)) bk = k
      if (shown.length && Math.abs(rowY(bestC, bk) - py) <= Math.max(gap, ((H - 16) / shown.length) * 0.6)) return { col: bestC, neuron: shown[bk], from: null }
    }
    // a line between two columns
    let best: GraphSel | null = null
    let bd = 6
    for (let c = 1; c < C; c++) {
      if (!columns[c].wired) continue
      if (px < colX(c - 1) || px > colX(c)) continue
      for (const ed of edges[c] ?? []) {
        const a = posOf(c - 1, ed.from)
        const b = posOf(c, ed.to)
        if (!a || !b) continue
        const vx = b.x - a.x
        const vy = b.y - a.y
        const tt = Math.max(0, Math.min(1, ((px - a.x) * vx + (py - a.y) * vy) / (vx * vx + vy * vy)))
        const d = Math.hypot(a.x + tt * vx - px, a.y + tt * vy - py)
        if (d < bd) {
          bd = d
          best = { col: c, neuron: ed.to, from: ed.from }
        }
      }
    }
    return best
  }

  const onKey = (e: KeyboardEvent) => {
    const cur = sel ?? { col: Math.min(1, C - 1), neuron: columns[Math.min(1, C - 1)].shown[0], from: null }
    const shown = columns[cur.col].shown
    const k = Math.max(0, shown.indexOf(cur.neuron))
    let next: GraphSel | null = null
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      const nk = Math.max(0, Math.min(shown.length - 1, k + (e.key === 'ArrowDown' ? 1 : -1)))
      next = { col: cur.col, neuron: shown[nk], from: null }
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      const nc = Math.max(0, Math.min(C - 1, cur.col + (e.key === 'ArrowRight' ? 1 : -1)))
      const ns = columns[nc].shown
      const nk = Math.min(ns.length - 1, Math.round((k / Math.max(1, shown.length - 1)) * (ns.length - 1)))
      next = { col: nc, neuron: ns[nk], from: null }
    } else if (e.key === 'Escape' && sel?.from != null) {
      next = { ...sel, from: null }
    }
    if (next) {
      e.preventDefault()
      onSelect(next)
    }
  }

  const tip = hover && (() => {
    const p = posOf(hover.from != null ? hover.col : hover.col, hover.neuron)
    if (!p) return null
    const column = columns[hover.col]
    const text =
      hover.from != null
        ? (() => {
            const ed = edges[hover.col]?.find((x) => x.from === hover.from && x.to === hover.neuron)
            return ed ? t('ex.mlp.tipEdge', { i: hover.from, j: hover.neuron, v: fx(ed.v, 4, true) }) : ''
          })()
        : `${column.label.split(' ')[0]} #${hover.neuron}: ${fx(column.values[hover.neuron], 4)}`
    const left = Math.min(Math.max(0, p.x - 70), width - 180)
    return (
      <div className="ex-graph__tip num" style={{ left, top: Math.max(0, p.y - 34) }} aria-hidden="true">
        {text}
      </div>
    )
  })()

  return (
    <div className="ex-graph" ref={wrap}>
      <div className="ex-graph__heads" aria-hidden="true">
        {columns.map((c, i) => (
          <span key={c.key} className="ex-graph__head num" style={{ left: `${colX(i)}px` }} data-filled={c.filled}>
            {width < 520 ? c.label.split(' ')[0] : c.label}
          </span>
        ))}
      </div>
      <div
        className="ex-graph__stage"
        role="application"
        tabIndex={0}
        aria-label={label}
        aria-roledescription={t('ex.mlp.graphRole')}
        onKeyDown={onKey}
        onPointerMove={(e) => e.pointerType === 'mouse' && setHover(hit(e))}
        onPointerLeave={() => setHover(null)}
        onClick={(e) => {
          const h = hit(e as unknown as PointerEvent<HTMLElement>)
          if (h) onSelect(h)
        }}
        style={{ height: H, cursor: hover ? 'pointer' : 'default' }}
      >
        <canvas ref={canvas} style={{ width: '100%', height: H }} />
        {tip}
      </div>
    </div>
  )
}
