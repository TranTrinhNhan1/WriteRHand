import { ArrowCounterClockwise } from '@phosphor-icons/react'
import { useEffect, useMemo, useRef } from 'react'
import { useI18n } from '../../i18n'
import type { MlpTrace } from '../../lib/trace'
import { Histogram } from '../Histogram'
import { HeatMap } from './HeatMap'

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
const clamp01 = (v: number) => Math.max(0, Math.min(1, v))
const CELL = 4 // size of one pixel of the picture before it is unrolled
const W = 1568 // logical canvas width: 2 units per pixel of the unrolled strip
const H = 124

/**
 * The 28 rows of the picture peel off one after another and line up end to end, which is exactly what the first layer
 * of the network does with the image: it reads 784 numbers in a row, with no idea which ones were neighbours.
 */
function Flatten({ pixels, play, onReplay }: { pixels: Float32Array; play: number; onReplay: () => void }) {
  const { t, lang } = useI18n()
  const ref = useRef<HTMLCanvasElement>(null)
  const latest = useRef(pixels)
  const progress = useRef(1)
  const raf = useRef(0)
  latest.current = pixels

  const draw = useRef((): void => undefined)
  draw.current = () => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, W, H)
    const stripW = W / 784
    const stripY = 70
    const stripH = 30
    const p = progress.current
    for (let r = 0; r < 28; r++) {
      const e = ease(clamp01(p * 1.7 - (r / 28) * 0.7))
      const cw = CELL + (stripW - CELL) * e
      const ch = CELL + (stripH - CELL) * e
      const x0 = r * 28 * stripW * e
      const y0 = r * CELL + (stripY - r * CELL) * e
      for (let c = 0; c < 28; c++) {
        const v = latest.current[r * 28 + c]
        ctx.fillStyle = `rgb(${Math.round(7 + 234 * v)},${Math.round(9 + 236 * v)},${Math.round(8 + 228 * v)})`
        ctx.fillRect(x0 + c * cw, y0, cw + 0.4, ch + 0.4)
      }
    }
    // ticks that mark where each original row starts in the strip
    ctx.fillStyle = 'rgba(198,241,53,0.85)'
    const a = ease(clamp01((p - 0.7) / 0.3))
    if (a > 0)
      for (let r = 0; r < 28; r++) {
        ctx.globalAlpha = a
        ctx.fillRect(r * 28 * stripW - 0.5, stripY + stripH + 4, 1.5, 8)
      }
    ctx.globalAlpha = 1
  }

  useEffect(() => {
    draw.current()
  }, [pixels])

  useEffect(() => {
    cancelAnimationFrame(raf.current)
    if (play === 0 || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      progress.current = 1
      draw.current()
      return
    }
    progress.current = 0
    const t0 = performance.now()
    const step = (now: number) => {
      progress.current = clamp01((now - t0) / 1800)
      draw.current()
      if (progress.current < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [play])

  return (
    <div className="flatten">
      <canvas ref={ref} width={W} height={H} role="img" aria-label={t('inside.mlp.flatAria')} lang={lang} />
      <button type="button" className="btn btn--quiet btn--small" onClick={onReplay}>
        <ArrowCounterClockwise size={16} aria-hidden="true" />
        {t('inside.replay')}
      </button>
    </div>
  )
}

export function MlpView({ trace, probs, pixels, play, onReplay }: { trace: MlpTrace; probs: ArrayLike<number>; pixels: Float32Array; play: number; onReplay: () => void }) {
  const { t } = useI18n()
  const [h1, h2] = trace.hidden
  const active = useMemo(() => trace.hidden.map((h) => h.reduce((n, v) => n + (v > 0 ? 1 : 0), 0)), [trace.hidden])
  const classes = useMemo(() => Array.from({ length: 10 }, (_, i) => i), [])
  const stages = [
    { key: 'in', op: null },
    { key: 'flat', op: 'inside.mlp.opFlat' },
    { key: 'h1', op: 'inside.mlp.op1' },
    { key: 'h2', op: 'inside.mlp.op2' },
    { key: 'out', op: 'inside.mlp.op3' },
  ] as const
  return (
    <ol className="flow">
      {stages.map((s, i) => (
        <li key={s.key} className="flow__stage" style={{ '--i': i } as React.CSSProperties}>
          {s.op && (
            <p className="flow__op">
              {t(s.op, s.key === 'h1' ? { a: 784, b: h1.length } : s.key === 'h2' ? { a: h1.length, b: h2.length } : s.key === 'out' ? { a: h2.length, b: 10 } : {})}
            </p>
          )}
          <div className="flow__card">
            <div className="flow__text">
              <h4 className="flow__title">{t(`inside.mlp.${s.key}.title`)}</h4>
              <p className="caption">
                {t(`inside.mlp.${s.key}.text`, s.key === 'h1' ? { n: active[0], of: h1.length } : s.key === 'h2' ? { n: active[1], of: h2.length } : {})}
              </p>
            </div>
            <div className="flow__viz">
              {s.key === 'in' && <HeatMap values={pixels} width={28} height={28} mode="gray" max={1} className="heat--input" label={t('inside.inputAria')} />}
              {s.key === 'flat' && <Flatten pixels={pixels} play={play} onReplay={onReplay} />}
              {s.key === 'h1' && <HeatMap values={h1} width={32} height={16} mode="active" label={t('inside.mlp.h1Aria', { n: active[0], of: h1.length })} />}
              {s.key === 'h2' && <HeatMap values={h2} width={16} height={16} mode="active" className="heat--narrow" label={t('inside.mlp.h2Aria', { n: active[1], of: h2.length })} />}
              {s.key === 'out' && <Histogram probs={probs} classes={classes} />}
            </div>
          </div>
        </li>
      ))}
    </ol>
  )
}
