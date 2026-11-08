import type { CSSProperties } from 'react'
import { useI18n } from '../i18n'
import { pct } from '../lib/format'
import { shortClassNames } from '../lib/selection'

interface Props {
  /** one value per class (probabilities, or vote shares), or null while there is no answer */
  values: ArrayLike<number> | null
  classes: readonly string[]
  /** what the values are, used in the accessible name of every bar ("probability", "vote share") */
  quantity: string
  /** draws the top bar in the warning colour (the model disagrees with the others) */
  dissent?: boolean
  /** a slim strip of vertical bars; long class names are labelled by their index (full names stay in the accessible text) */
  mini?: boolean
}

/**
 * One bar per class, always in class order so the shape of the models' doubt can be compared
 * between cards. Single-character labels (digits) fit under vertical bars; longer class names
 * (Fashion-MNIST) get a horizontal row each, so every bar carries the full class name.
 */
export function ClassBars({ values, classes, quantity, dissent, mini }: Props) {
  const { t } = useI18n()
  let top = -1
  if (values) for (let i = 0; i < classes.length; i++) if (top < 0 || values[i] > values[top]) top = i
  const short = shortClassNames(classes)
  const vertical = short || !!mini
  return (
    <ul className={`cbars ${vertical ? 'cbars--v' : 'cbars--h'}${mini ? ' cbars--mini' : ''}${dissent ? ' is-dissent' : ''}${values ? '' : ' is-empty'}`} aria-label={t('rec.bars.aria', { quantity })}>
      {classes.map((name, i) => {
        const v = values ? Math.max(0, Math.min(1, values[i])) : 0
        const style = { '--d': `${i * 22}ms`, '--v': v } as CSSProperties
        return (
          <li key={`${i}-${name}`} className={i === top ? 'cbars__row is-top' : 'cbars__row'} style={style}>
            <span className="cbars__name" aria-hidden="true" title={name}>
              {short ? name : i}
            </span>
            <span className="cbars__track" aria-hidden="true">
              <span className="cbars__fill" />
            </span>
            {!vertical && (
              <span className="cbars__val num" aria-hidden="true">
                {values ? pct(v, 0) : ''}
              </span>
            )}
            <span className="visually-hidden">{t('rec.bars.row', { name, value: values ? pct(v, 1) : t('common.na') })}</span>
          </li>
        )
      })}
    </ul>
  )
}
