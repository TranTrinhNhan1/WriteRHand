import { useId, type CSSProperties } from 'react'

interface Option<T extends string> {
  value: T
  label: string
}

interface Props<T extends string> {
  legend: string
  value: T
  options: Option<T>[]
  onChange: (v: T) => void
  hideLegend?: boolean
}

/**
 * Native radio inputs styled as a joined control, so arrow keys and screen readers work as expected.
 * Short sets (up to three options) get equal columns and a thumb that slides to the chosen one.
 * Longer sets wrap like chips and just change colour.
 */
export function Segmented<T extends string>({ legend, value, options, onChange, hideLegend }: Props<T>) {
  const name = useId()
  const slide = options.length <= 3
  const index = Math.max(0, options.findIndex((o) => o.value === value))
  const style = { '--i': index, '--n': options.length } as CSSProperties
  return (
    <fieldset className={`seg${slide ? ' seg--slide' : ''}`} style={style}>
      <legend className={hideLegend ? 'visually-hidden' : 'seg__legend'}>{legend}</legend>
      <div className="seg__row">
        {slide && <span className="seg__thumb" aria-hidden="true" />}
        {options.map((o) => (
          <label key={o.value} className="seg__opt">
            <input type="radio" name={name} value={o.value} checked={o.value === value} onChange={() => onChange(o.value)} />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}
