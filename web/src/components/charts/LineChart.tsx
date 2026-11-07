import { useId, useMemo, useState } from 'react'
import { useI18n } from '../../i18n'
import { useElementWidth } from '../../lib/hooks'
import { finite, linearTicks, logTicks, makeScale, tickLabel } from '../../lib/scale'
import { Marker, type MarkerKind } from './ChartFigure'

export interface LineSeries {
  key: string
  label: string
  color: string
  marker?: MarkerKind
  dashed?: boolean
  points: { x: number; y: number }[]
}

interface Props {
  series: LineSeries[]
  xLabel: string
  yLabel: string
  xScale?: 'linear' | 'log'
  /** Categorical x axis: labels for the x positions (x values are the indices). */
  xCategories?: string[]
  xFormat?: (x: number) => string
  yFormat: (y: number) => string
  yMin?: number
  yMax?: number
  /** Vertical reference marks, e.g. the chosen hyper-parameter value. */
  marks?: { x: number; label: string }[]
  ariaLabel: string
  height?: number
}

const M = { top: 14, right: 18, bottom: 54, left: 66 }

export function LineChart({ series, xLabel, yLabel, xScale = 'linear', xCategories, xFormat, yFormat, yMin, yMax, marks = [], ariaLabel, height = 280 }: Props) {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const { t } = useI18n()
  const titleId = useId()
  const [cursor, setCursor] = useState<number | null>(null)

  const clean = useMemo(
    () => series.map((s) => ({ ...s, points: s.points.filter((p) => finite(p.x) && finite(p.y) && (xScale !== 'log' || p.x > 0)) })),
    [series, xScale],
  )
  const xs = useMemo(() => [...new Set(clean.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b), [clean])
  const ys = clean.flatMap((s) => s.points.map((p) => p.y))
  if (xs.length === 0) return <p className="caption">{t('chart.noPlot')}</p>

  const lo = yMin ?? Math.min(...ys)
  const hi = yMax ?? Math.max(...ys)
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08
  const yTicks = linearTicks(yMin ?? lo - pad, yMax ?? hi + pad, 5)
  const yDomain: [number, number] = [yTicks[0], yTicks[yTicks.length - 1]]

  const xMin = xs[0]
  const xMax = xs[xs.length - 1]
  let xTicks: number[]
  let xDomain: [number, number]
  if (xCategories) {
    xTicks = xs
    xDomain = [xMin - 0.4, xMax + 0.4]
  } else if (xScale === 'log') {
    xTicks = logTicks(xMin, xMax)
    xDomain = [xTicks[0], xTicks[xTicks.length - 1]]
  } else {
    const span = xMax - xMin
    const padX = span === 0 ? 1 : span * 0.03
    xDomain = [xMin - padX, xMax + padX]
    const target = Math.max(2, Math.min(8, Math.floor(width / 70)))
    xTicks = span === 0 ? [xMin] : linearTicks(xMin, xMax, target).filter((t) => t >= xMin - span * 1e-9 && t <= xMax + span * 1e-9)
    if (xs.every(Number.isInteger)) xTicks = xTicks.filter(Number.isInteger)
    if (xTicks.length < 2 && span > 0) xTicks = [xMin, xMax]
  }

  const plotW = Math.max(100, width - M.left - M.right)
  const plotH = height - M.top - M.bottom
  const sx = makeScale(xCategories ? 'linear' : xScale, xDomain, [M.left, M.left + plotW])
  const sy = makeScale('linear', yDomain, [M.top + plotH, M.top])
  const fmtX = xFormat ?? ((v: number) => (xCategories ? (xCategories[v] ?? String(v)) : tickLabel(v)))

  const move = (clientX: number, rect: DOMRect) => {
    const px = clientX - rect.left
    let best = xs[0]
    for (const v of xs) if (Math.abs(sx(v) - px) < Math.abs(sx(best) - px)) best = v
    setCursor(best)
  }

  const idx = cursor == null ? -1 : xs.indexOf(cursor)
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault()
      const next = Math.min(xs.length - 1, Math.max(0, (idx < 0 ? (e.key === 'ArrowRight' ? -1 : xs.length) : idx) + (e.key === 'ArrowRight' ? 1 : -1)))
      setCursor(xs[next])
    } else if (e.key === 'Escape') setCursor(null)
  }

  const tipRows = cursor == null ? [] : clean.map((s) => ({ s, p: s.points.find((p) => p.x === cursor) })).filter((r) => r.p)
  const tipW = 176
  const rightSpace = cursor == null ? 0 : M.left + plotW - sx(cursor) - 12
  const tipLeft = cursor == null ? 0 : rightSpace >= tipW ? sx(cursor) + 12 : Math.max(M.left, sx(cursor) - 12 - tipW)

  return (
    <div ref={ref} className="chart__svg">
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="group"
          aria-labelledby={titleId}
          tabIndex={0}
          className="line-chart"
          onKeyDown={onKey}
          onBlur={() => setCursor(null)}
          onPointerMove={(e) => move(e.clientX, e.currentTarget.getBoundingClientRect())}
          onPointerLeave={() => setCursor(null)}
        >
          <title id={titleId}>{`${ariaLabel}. ${t('chart.keys')}`}</title>
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={M.left + plotW} y1={sy(t)} y2={sy(t)} className="grid" />
              <text x={M.left - 8} y={sy(t) + 4} textAnchor="end" className="axis-text">
                {yFormat(t)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={M.top + plotH} y2={M.top + plotH + 4} className="tick" />
              <text x={sx(t)} y={M.top + plotH + 18} textAnchor="middle" className="axis-text">
                {fmtX(t)}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={M.left + plotW} y1={M.top + plotH} y2={M.top + plotH} className="axis" />
          <text x={M.left + plotW / 2} y={height - 10} textAnchor="middle" className="axis-title">
            {xLabel}
          </text>
          <text transform={`translate(12 ${M.top + plotH / 2}) rotate(-90)`} textAnchor="middle" className="axis-title">
            {yLabel}
          </text>

          {marks.map((m) => (
            <g key={m.x + m.label}>
              <line x1={sx(m.x)} x2={sx(m.x)} y1={M.top} y2={M.top + plotH} className="mark-line" />
              <text x={sx(m.x) + (sx(m.x) > M.left + plotW - 70 ? -6 : 6)} y={M.top + 11} textAnchor={sx(m.x) > M.left + plotW - 70 ? 'end' : 'start'} className="mark-text">
                {m.label}
              </text>
            </g>
          ))}

          {clean.map((s) => (
            <g key={s.key}>
              <polyline
                fill="none"
                points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')}
                style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dashed ? '5 4' : undefined, strokeLinejoin: 'round' }}
              />
              {s.points.length <= 24 &&
                s.points.map((p) => <Marker key={p.x} kind={s.marker ?? 'circle'} cx={sx(p.x)} cy={sy(p.y)} r={3.5} color={s.color} />)}
            </g>
          ))}

          {cursor != null && (
            <g pointerEvents="none">
              <line x1={sx(cursor)} x2={sx(cursor)} y1={M.top} y2={M.top + plotH} className="cursor" />
              {tipRows.map(({ s, p }) => (
                <Marker key={s.key} kind={s.marker ?? 'circle'} cx={sx(cursor)} cy={sy(p!.y)} r={5} color={s.color} />
              ))}
              <g transform={`translate(${tipLeft} ${M.top + 4})`}>
                <rect width={tipW} height={22 + tipRows.length * 18} rx={6} className="tip-box" />
                <text x={10} y={16} className="tip-title">
                  {t('chart.xValue', { label: xLabel.split(' (')[0], value: fmtX(cursor) })}
                </text>
                {tipRows.map(({ s, p }, i) => (
                  <g key={s.key} transform={`translate(10 ${32 + i * 18})`}>
                    <rect y={-9} width={9} height={9} rx={2} style={{ fill: s.color }} />
                    <text x={15} className="tip-text">
                      {`${s.label}: ${yFormat(p!.y)}`}
                    </text>
                  </g>
                ))}
              </g>
            </g>
          )}
        </svg>
      )}
    </div>
  )
}
