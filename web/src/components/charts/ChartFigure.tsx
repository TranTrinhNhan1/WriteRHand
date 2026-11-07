import type { ReactNode } from 'react'
import { useI18n } from '../../i18n'

export type MarkerKind = 'circle' | 'square' | 'diamond'

export interface LegendItem {
  label: string
  color: string
  marker?: MarkerKind
  dashed?: boolean
}

export interface TableData {
  columns: string[]
  rows: (string | number)[][]
}

export function Marker({ kind = 'square', cx, cy, r = 4, color, filled = true }: { kind?: MarkerKind; cx: number; cy: number; r?: number; color: string; filled?: boolean }) {
  const style = { fill: filled ? color : 'var(--surface)', stroke: color, strokeWidth: 2 }
  if (kind === 'circle') return <circle cx={cx} cy={cy} r={r} style={style} />
  if (kind === 'diamond') return <path d={`M${cx} ${cy - r - 1}L${cx + r + 1} ${cy}L${cx} ${cy + r + 1}L${cx - r - 1} ${cy}Z`} style={style} />
  return <rect x={cx - r + 0.5} y={cy - r + 0.5} width={2 * r - 1} height={2 * r - 1} rx={1} style={style} />
}

function LegendSwatch({ item }: { item: LegendItem }) {
  return (
    <svg width="22" height="12" viewBox="0 0 22 12" aria-hidden="true">
      {item.marker && item.marker !== 'square' ? (
        <>
          <line x1="0" x2="22" y1="6" y2="6" style={{ stroke: item.color, strokeWidth: 2, strokeDasharray: item.dashed ? '4 3' : undefined }} />
          <Marker kind={item.marker} cx={11} cy={6} r={3.5} color={item.color} />
        </>
      ) : (
        <rect x="5" y="1" width="12" height="10" rx="2" style={{ fill: item.color }} />
      )}
    </svg>
  )
}

interface Props {
  title: string
  caption?: ReactNode
  legend?: LegendItem[]
  table?: TableData
  tableLabel?: string
  children: ReactNode
}

/** A chart with a title, an always-visible legend and a table alternative for every plotted value. */
export function ChartFigure({ title, caption, legend, table, tableLabel, children }: Props) {
  const { t } = useI18n()
  return (
    <figure className="chart">
      <figcaption className="chart__head">
        <h3 className="chart__title">{title}</h3>
        {caption && <p className="caption">{caption}</p>}
      </figcaption>
      {legend && legend.length > 0 && (
        <ul className="legend" aria-label={t('common.legend')}>
          {legend.map((l) => (
            <li key={l.label}>
              <LegendSwatch item={l} />
              {l.label}
            </li>
          ))}
        </ul>
      )}
      {children}
      {table && (
        <details className="chart__data">
          <summary>{t('common.dataTable')}</summary>
          <div className="table-wrap" role="region" aria-label={tableLabel ?? title} tabIndex={0}>
            <table className="table num">
              <thead>
                <tr>
                  {table.columns.map((c, i) => (
                    <th key={c + i} scope="col" className={i === 0 ? 'is-left' : undefined}>
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((r, i) => (
                  <tr key={i}>
                    {r.map((cell, j) => (j === 0 ? <th key={j} scope="row" className="is-left">{cell}</th> : <td key={j}>{cell}</td>))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </figure>
  )
}
