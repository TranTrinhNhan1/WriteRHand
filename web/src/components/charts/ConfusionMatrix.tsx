import { useEffect, useId, useMemo, useState } from 'react'
import { useI18n } from '../../i18n'
import { int, pct } from '../../lib/format'
import { useElementWidth } from '../../lib/hooks'
import { topConfusions } from '../../lib/results'
import { cellOpacity } from '../../lib/scale'
import { shortClassNames } from '../../lib/selection'
import { Segmented } from '../Segmented'

interface Props {
  matrix: number[][]
  /** display names of the classes (index = row / column) */
  classes: readonly string[]
  /** Used in the accessible name, e.g. "MLP, test split". */
  name: string
}

/** Axis label that fits: the class name when it is short, else its first characters (the full name is in a tooltip and the key below). */
export function axisLabel(name: string, max = 5): string {
  return name.length <= max ? name : `${name.slice(0, max - 1).trimEnd()}…`
}

export function ConfusionMatrix({ matrix, classes, name }: Props) {
  const { t } = useI18n()
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const titleId = useId()
  const [view, setView] = useState<'all' | 'errors'>('all')
  const [sel, setSel] = useState<{ r: number; c: number } | null>(null)
  const n = matrix.length
  const short = shortClassNames(classes)
  // long names: row labels get room for a few characters, column labels are turned 50 degrees
  const LEFT = short ? 34 : 74
  const TOP = short ? 44 : 70

  useEffect(() => setSel(null), [matrix])

  const rowTotals = useMemo(() => matrix.map((r) => r.reduce((a, b) => a + b, 0)), [matrix])
  const maxOff = useMemo(() => Math.max(1e-9, ...matrix.flatMap((row, r) => row.map((v, c) => (r === c ? 0 : rowTotals[r] ? v / rowTotals[r] : 0)))), [matrix, rowTotals])
  const top = useMemo(() => topConfusions(matrix, 5), [matrix])
  const isTop = (r: number, c: number) => top.some((t) => t.trueDigit === r && t.predDigit === c)

  const RIGHT = short ? 0 : 30
  const side = Math.max(240, Math.min(width - RIGHT, 600))
  const cell = (side - LEFT) / n
  const showCount = cell >= 27
  const fontSize = Math.max(9, Math.min(13, cell * 0.36))

  const share = (r: number, c: number) => (rowTotals[r] ? matrix[r][c] / rowTotals[r] : 0)
  const strength = (r: number, c: number) => {
    const s = share(r, c)
    if (view === 'errors') return r === c ? 0 : s / maxOff
    return s
  }

  const cur = sel ?? (top[0] ? { r: top[0].trueDigit, c: top[0].predDigit } : { r: 0, c: 0 })
  const onKey = (e: React.KeyboardEvent) => {
    const d: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }
    const step = d[e.key]
    if (!step) return
    e.preventDefault()
    setSel({ r: Math.min(n - 1, Math.max(0, cur.r + step[0])), c: Math.min(n - 1, Math.max(0, cur.c + step[1])) })
  }

  const count = matrix[cur.r]?.[cur.c] ?? 0
  const readout = t(cur.r === cur.c ? 'cm.readoutOk' : 'cm.readoutWrong', {
    t: classes[cur.r],
    p: classes[cur.c],
    count: int(count),
    total: int(rowTotals[cur.r]),
    share: pct(share(cur.r, cur.c), 1),
  })

  return (
    <div className="cm">
      <div className="cm__controls">
        <Segmented
          legend={t('cm.colour')}
          value={view}
          onChange={setView}
          options={[
            { value: 'all', label: t('cm.all') },
            { value: 'errors', label: t('cm.errors') },
          ]}
        />
        <p className="caption cm__scale">
          {view === 'all' ? t('cm.scaleAll') : t('cm.scaleErrors')}
        </p>
      </div>

      <div ref={ref} className="cm__plot">
        {width > 0 && (
          <svg
            width={side + RIGHT}
            height={side + TOP - LEFT}
            role="group"
            aria-labelledby={titleId}
            tabIndex={0}
            className="cm__svg"
            onKeyDown={onKey}
          >
            <title id={titleId}>{t('cm.aria', { name })}</title>
            <text x={LEFT + (cell * n) / 2} y={13} textAnchor="middle" className="axis-title">
              {t('cm.predicted')}
            </text>
            <text transform={`translate(11 ${TOP + (cell * n) / 2}) rotate(-90)`} textAnchor="middle" className="axis-title">
              {t('cm.truth')}
            </text>
            {classes.map((c, i) => (
              <g key={`${i}-${c}`}>
                {short ? (
                  <text x={LEFT + i * cell + cell / 2} y={TOP - 8} textAnchor="middle" className="axis-text">
                    {c}
                  </text>
                ) : (
                  <text transform={`translate(${LEFT + i * cell + cell / 2 + 3} ${TOP - 6}) rotate(-50)`} textAnchor="start" className="axis-text">
                    <title>{c}</title>
                    {axisLabel(c, 8)}
                  </text>
                )}
                <text x={LEFT - 8} y={TOP + i * cell + cell / 2 + 4} textAnchor="end" className="axis-text">
                  {!short && <title>{c}</title>}
                  {short ? c : axisLabel(c, 8)}
                </text>
              </g>
            ))}
            {matrix.map((row, r) =>
              row.map((v, c) => {
                const s = strength(r, c)
                const op = cellOpacity(s)
                const muted = view === 'errors' && r === c
                const dark = op > 0.72 && !muted
                const x = LEFT + c * cell
                const y = TOP + r * cell
                return (
                  <g key={`${r}-${c}`} onPointerEnter={() => setSel({ r, c })} onClick={() => setSel({ r, c })}>
                    <title>{t('cm.cell', { t: classes[r], p: classes[c], count: int(v), share: pct(share(r, c), 1) })}</title>
                    <rect x={x + 0.5} y={y + 0.5} width={cell - 1} height={cell - 1} rx={2} className="cm__cell" />
                    {!muted && <rect x={x + 0.5} y={y + 0.5} width={cell - 1} height={cell - 1} rx={2} className="cm__fill" style={{ opacity: op }} />}
                    {showCount && v > 0 && (
                      <text x={x + cell / 2} y={y + cell / 2 + fontSize * 0.35} textAnchor="middle" fontSize={fontSize} className={dark ? 'cm__num is-on-dark' : 'cm__num'}>
                        {v}
                      </text>
                    )}
                  </g>
                )
              }),
            )}
            {matrix.map((row, r) =>
              row.map((_, c) =>
                isTop(r, c) ? (
                  <rect key={`t${r}-${c}`} x={LEFT + c * cell + 1.5} y={TOP + r * cell + 1.5} width={cell - 3} height={cell - 3} rx={2} className="cm__top" pointerEvents="none" />
                ) : null,
              ),
            )}
            <rect x={LEFT + cur.c * cell + 1} y={TOP + cur.r * cell + 1} width={cell - 2} height={cell - 2} rx={2} className="cm__sel" pointerEvents="none" />
          </svg>
        )}
      </div>

      <p className="cm__readout" role="status" aria-live="polite">
        {readout}
      </p>

      {!short && classes.some((c) => axisLabel(c, 8) !== c) && (
        <div className="cm__key">
          <p className="label">{t('cm.key')}</p>
          <ol className="cm__keylist">
            {classes.map((c, i) => axisLabel(c, 8) === c ? null : (
              <li key={`${i}-${c}`}>
                <span className="num">{axisLabel(c, 8)}</span>
                <span>{c}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      <div className="cm__top-list">
        <p className="label">{t('cm.pairs')}</p>
        {top.length === 0 ? (
          <p className="caption">{t('cm.noErrors')}</p>
        ) : (
          <ul className="pairs">
            {top.map((pr) => (
              <li key={`${pr.trueDigit}-${pr.predDigit}`}>
                <button type="button" className="pair" onClick={() => setSel({ r: pr.trueDigit, c: pr.predDigit })}
                  aria-pressed={cur.r === pr.trueDigit && cur.c === pr.predDigit}
                  aria-label={t('cm.pairAria', { t: classes[pr.trueDigit], p: classes[pr.predDigit], count: int(pr.count), share: pct(pr.share, 1) })}
                >
                  <span className="pair__digits">
                    {classes[pr.trueDigit]} <span aria-hidden="true">{'→'}</span> {classes[pr.predDigit]}
                  </span>
                  <span className="num pair__count">{int(pr.count)}</span>
                  <span className="num pair__share">{pct(pr.share, 1)}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
        <p className="caption">{t('cm.pairsNote')}</p>
      </div>

      <details className="chart__data">
        <summary>{t('cm.table')}</summary>
        <div className="table-wrap" role="region" aria-label={t('cm.region', { name })} tabIndex={0}>
          <table className="table num table--matrix">
            <caption className="visually-hidden">{t('cm.tableCaption')}</caption>
            <thead>
              <tr>
                <th scope="col" className="is-left">
                  {t('cm.corner')}
                </th>
                {classes.map((c, i) => (
                  <th key={`${i}-${c}`} scope="col">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, r) => (
                <tr key={r}>
                  <th scope="row" className="is-left">
                    {classes[r]}
                  </th>
                  {row.map((v, c) => (
                    <td key={c} className={r === c ? 'is-diag' : undefined}>
                      {int(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
