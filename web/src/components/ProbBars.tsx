import { useI18n } from '../i18n'
import { pct } from '../lib/format'

interface Props {
  probs: ArrayLike<number> | null
  classes: number[]
  mode: 'top3' | 'all'
  stale?: boolean
}

/**
 * Horizontal probability bars. "Top 3" sorts by value; "All 10" keeps digit order so rows do not jump.
 * The fill is a scaled element, so the CSS spring only moves transform.
 */
export function ProbBars({ probs, classes, mode, stale }: Props) {
  const { t } = useI18n()
  const rows = classes.map((c, i) => ({ c, p: probs ? probs[i] : 0 }))
  const best = probs ? rows.reduce((a, b) => (b.p > a.p ? b : a)).c : null
  const shown = mode === 'top3' ? [...rows].sort((a, b) => b.p - a.p).slice(0, 3) : rows
  return (
    <ul className={`bars${stale ? ' is-stale' : ''}`} aria-label={t('rec.bars.aria')}>
      {shown.map(({ c, p }, i) => (
        <li key={c} className={c === best ? 'bars__row is-top' : 'bars__row'} style={{ '--d': `${i * 28}ms` } as React.CSSProperties}>
          <span className="bars__digit">{c}</span>
          <span className="bars__track" aria-hidden="true">
            <span className="bars__fill" style={{ transform: `scaleX(${Math.max(0, Math.min(1, p))})` }} />
          </span>
          <span className="bars__value">{probs ? pct(p, 1) : '–'}</span>
          <span className="visually-hidden">{t('rec.bars.row', { digit: c, value: pct(p, 1) })}</span>
        </li>
      ))}
    </ul>
  )
}
