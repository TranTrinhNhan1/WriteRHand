import { useId, useState } from 'react'
import { useI18n } from '../../i18n'
import { useElementWidth } from '../../lib/hooks'
import { finite, linearTicks, logTicks, makeScale, tickLabel } from '../../lib/scale'

export interface BarSeries {
  label: string
  color: string
}

export interface BarRow {
  label: string
  values: (number | null | undefined)[]
}

interface Props {
  rows: BarRow[]
  series: BarSeries[]
  scale: 'linear' | 'log'
  axisLabel: string
  format: (v: number) => string
  ariaLabel: string
  /** Linear axes start at this value (default 0: bars encode length from a zero baseline). */
  min?: number
  max?: number
}

const BAR_H = 13
const BAR_GAP = 3
const ROW_LABEL_H = 22
const ROW_GAP = 16
const AXIS_H = 52
const VALUE_W = 74

/** Horizontal grouped bars: category labels sit above each group, so nothing is ever truncated at 360 px. */
export function BarChart({ rows, series, scale, axisLabel, format, ariaLabel, min = 0, max }: Props) {
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const { t } = useI18n()
  const titleId = useId()
  const [hover, setHover] = useState<string | null>(null)

  const all = rows.flatMap((r) => r.values).filter(finite)
  const positive = all.filter((v) => v > 0)
  let domain: [number, number]
  let ticks: number[]
  if (scale === 'log') {
    const lo = positive.length ? Math.min(...positive) : 1
    const hi = positive.length ? Math.max(...positive) : 10
    // Start one step below the smallest value so that its bar is still visible.
    ticks = logTicks(lo / 5, hi)
    domain = [ticks[0], ticks[ticks.length - 1]]
  } else {
    const hi = max ?? (all.length ? Math.max(...all) : 1)
    ticks = linearTicks(min, hi, 5)
    domain = [Math.min(min, ticks[0]), ticks[ticks.length - 1]]
  }

  const left = 0
  const plotW = Math.max(120, width - left - VALUE_W)
  const x = makeScale(scale, domain, [left, left + plotW])
  const groupH = series.length * BAR_H + (series.length - 1) * BAR_GAP
  const rowH = ROW_LABEL_H + groupH
  const height = rows.length * rowH + (rows.length - 1) * ROW_GAP + AXIS_H
  const plotBottom = height - AXIS_H + 6

  return (
    <div ref={ref} className="chart__svg">
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-labelledby={titleId} className="bar-chart">
          <title id={titleId}>{ariaLabel}</title>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={0} y2={plotBottom} className="grid" />
              <text x={x(t)} y={plotBottom + 16} textAnchor="middle" className="axis-text">
                {tickLabel(t)}
              </text>
            </g>
          ))}
          <text x={left + plotW / 2} y={plotBottom + 40} textAnchor="middle" className="axis-title">
            {axisLabel}
          </text>
          {rows.map((r, i) => {
            const top = i * (rowH + ROW_GAP)
            return (
              <g key={r.label} transform={`translate(0 ${top})`}>
                <text x={0} y={14} className="row-label">
                  {r.label}
                </text>
                {r.values.map((v, k) => {
                  const y = ROW_LABEL_H + k * (BAR_H + BAR_GAP)
                  const key = `${i}-${k}`
                  if (!finite(v) || (scale === 'log' && v <= 0)) {
                    return (
                      <text key={k} x={x(domain[0]) + 6} y={y + BAR_H - 2} className="value-text is-muted">
                        {t('chart.noData')}
                      </text>
                    )
                  }
                  const x1 = Math.max(x(v), x(domain[0]) + 2)
                  return (
                    <g
                      key={k}
                      onPointerEnter={() => setHover(key)}
                      onPointerLeave={() => setHover(null)}
                      opacity={hover && hover !== key ? 0.55 : 1}
                    >
                      <title>{t('chart.barTitle', { row: r.label, series: series[k].label, value: format(v) })}</title>
                      <rect x={x(domain[0])} y={y} width={x1 - x(domain[0])} height={BAR_H} rx={2} style={{ fill: series[k].color }} />
                      <text x={x1 + 8} y={y + BAR_H - 2} className="value-text">
                        {format(v)}
                      </text>
                    </g>
                  )
                })}
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}
