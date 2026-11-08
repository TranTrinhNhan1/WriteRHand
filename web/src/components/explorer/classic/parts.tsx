import { useRef, type KeyboardEvent, type ReactNode } from 'react'
import { useI18n } from '../../../i18n'
import { HeatMap, type HeatMode } from '../HeatMap'
import { fmt, fmtDiff } from './model'

/** Class chooser: one thumbnail per class (a real 28x28 parameter map) with the class name. Arrow keys move. */
export function ClassPicker({
  classes,
  value,
  onChange,
  thumb,
  sub,
  label,
  width,
  height,
}: {
  classes: string[]
  value: number
  onChange: (k: number) => void
  thumb: (k: number) => { values: ArrayLike<number>; mode: HeatMode; max?: number }
  sub?: (k: number) => string
  label: string
  width: number
  height: number
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const onKey = (e: KeyboardEvent) => {
    const d: Record<string, number> = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }
    let n = value
    if (d[e.key]) n = (value + d[e.key] + classes.length) % classes.length
    else if (e.key === 'Home') n = 0
    else if (e.key === 'End') n = classes.length - 1
    else return
    e.preventDefault()
    onChange(n)
    refs.current[n]?.focus()
  }
  return (
    <div className="exc-picker" role="radiogroup" aria-label={label} onKeyDown={onKey}>
      {classes.map((name, k) => {
        const th = thumb(k)
        return (
          <button
            key={k}
            ref={(el) => {
              refs.current[k] = el
            }}
            type="button"
            role="radio"
            aria-checked={k === value}
            tabIndex={k === value ? 0 : -1}
            className="exc-picker__opt"
            onClick={() => onChange(k)}
          >
            <span className="exc-picker__map" aria-hidden="true">
              <HeatMap values={th.values} width={width} height={height} mode={th.mode} max={th.max} label="" />
            </span>
            <span className="exc-picker__name" title={name}>
              {name}
            </span>
            {sub && <span className="exc-picker__sub num">{sub(k)}</span>}
          </button>
        )
      })}
    </div>
  )
}

/** A labelled 28x28 map that selects pixels on click / hover / arrow keys. */
export function MapFigure({ title, caption, children }: { title: string; caption?: ReactNode; children: ReactNode }) {
  return (
    <figure className="exc-fig">
      <figcaption className="exc-fig__title">{title}</figcaption>
      {children}
      {caption && <p className="exc-fig__cap">{caption}</p>}
    </figure>
  )
}

/** Definition-list readout: symbol, meaning, value. */
export function Readout({ rows }: { rows: { sym: ReactNode; label: string; value: string; strong?: boolean }[] }) {
  return (
    <dl className="exc-readout">
      {rows.map((r, i) => (
        <div key={i} className={r.strong ? 'exc-readout__row is-strong' : 'exc-readout__row'}>
          <dt>
            <span className="exc-readout__sym">{r.sym}</span>
            <span className="exc-readout__label">{r.label}</span>
          </dt>
          <dd className="num">{r.value}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Rebuilt value vs the value returned by the deployed graph, per class, with |difference|. */
export function VerifyTable({ classes, rebuilt, runtime, title, note, formula }: { classes: string[]; rebuilt: ArrayLike<number>; runtime: ArrayLike<number>; title: string; note: string; formula: string }) {
  const { t } = useI18n()
  let worst = 0
  let size = 0
  for (let k = 0; k < classes.length; k++) {
    worst = Math.max(worst, Math.abs(rebuilt[k] - runtime[k]))
    size = Math.max(size, Math.abs(runtime[k]))
  }
  return (
    <section className="exc-verify" aria-label={title}>
      <h4 className="exc-h">{title}</h4>
      <p className="exc-verify__sum num">{t('ex.classic.verifyMax', { d: fmtDiff(worst), n: classes.length, rel: fmtDiff(size > 0 ? worst / size : 0) })}</p>
      <p className="exc-fig__cap">{formula}</p>
      <div className="exc-verify__scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">{t('ex.classic.class')}</th>
              <th scope="col">{t('ex.classic.rebuilt')}</th>
              <th scope="col">{t('ex.classic.graph')}</th>
              <th scope="col">|Δ|</th>
            </tr>
          </thead>
          <tbody>
            {classes.map((name, k) => (
              <tr key={k}>
                <th scope="row">{name}</th>
                <td className="num">{fmt(rebuilt[k], 4)}</td>
                <td className="num">{fmt(runtime[k], 4)}</td>
                <td className="num">{fmtDiff(Math.abs(rebuilt[k] - runtime[k]))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note note--muted">{note}</p>
    </section>
  )
}

/** Visually hidden live region for step narration. */
export function Narration({ text }: { text: string }) {
  return (
    <p className="visually-hidden" aria-live="polite" aria-atomic="true">
      {text}
    </p>
  )
}

export function Legend({ items }: { items: { swatch: 'pos' | 'neg' | 'accent' | 'ink' | 'mute' | 'mark'; text: string }[] }) {
  return (
    <ul className="exc-legend">
      {items.map((i) => (
        <li key={i.text}>
          <i className={`exc-sw exc-sw--${i.swatch}`} aria-hidden="true" />
          {i.text}
        </li>
      ))}
    </ul>
  )
}
