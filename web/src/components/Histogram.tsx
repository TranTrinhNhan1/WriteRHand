import { useI18n } from '../i18n'
import { pct } from '../lib/format'

interface Props {
  probs: ArrayLike<number> | null
  classes: number[]
  stale?: boolean
}

/**
 * Ten vertical bars, one per digit, always in digit order so the shape of a model's doubt is easy to compare
 * between cards. The bar is a scaled element, so the CSS spring only moves transform.
 */
export function Histogram({ probs, classes, stale }: Props) {
  const { t } = useI18n()
  const best = probs ? classes.reduce((a, c, i) => (probs[i] > probs[classes.indexOf(a)] ? c : a), classes[0]) : null
  return (
    <ul className={`hist${stale ? ' is-stale' : ''}`} aria-label={t('rec.card.hist')}>
      {classes.map((c, i) => {
        const p = probs ? Math.max(0, Math.min(1, probs[i])) : 0
        return (
          <li key={c} className={c === best ? 'hist__col is-top' : 'hist__col'} style={{ '--d': `${i * 22}ms` } as React.CSSProperties}>
            <span className="hist__bar" aria-hidden="true">
              <span className="hist__fill" style={{ transform: `scaleY(${p})` }} />
            </span>
            <span className="hist__digit" aria-hidden="true">
              {c}
            </span>
            <span className="visually-hidden">{t('rec.bars.row', { digit: c, value: pct(p, 1) })}</span>
          </li>
        )
      })}
    </ul>
  )
}
