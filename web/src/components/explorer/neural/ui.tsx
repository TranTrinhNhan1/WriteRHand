/** Small building blocks shared by the MLP and CNN views (and the replay panel). */
import { CheckCircle, Info, WarningCircle } from '@phosphor-icons/react'
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import { useI18n } from '../../../i18n'
import { softmax } from '../../../engine/reconstruct'
import { HeatMap } from '../HeatMap'
import { diff, fx, shape } from './format'

// --------------------------------------------------------------------------- state that survives a new drawing
/**
 * The Explorer unmounts a view while a new input is being computed (it may never show the old
 * result). Selections live in this per-model store so they survive that pause.
 */
export const PersistContext = createContext<Map<string, unknown> | null>(null)

export function usePersisted<T>(name: string, init: T): [T, (v: T | ((prev: T) => T)) => void] {
  const store = useContext(PersistContext)
  const [v, setV] = useState<T>(() => (store?.has(name) ? (store.get(name) as T) : init))
  const set = useCallback(
    (nv: T | ((prev: T) => T)) =>
      setV((prev) => {
        const r = typeof nv === 'function' ? (nv as (p: T) => T)(prev) : nv
        store?.set(name, r)
        return r
      }),
    [store, name],
  )
  return [v, set]
}

// --------------------------------------------------------------------------- rebuilt vs returned
/** Tolerance under which a float32 graph value and a float64 reconstruction count as equal. */
export const AGREE = 1e-3

export function CheckRow({ label, rebuilt, runtime }: { label: ReactNode; rebuilt: number; runtime: number }) {
  const { t } = useI18n()
  const d = Math.abs(rebuilt - runtime)
  const ok = d <= AGREE * Math.max(1, Math.abs(runtime))
  return (
    <div className="ex-check" data-ok={ok}>
      <span className="ex-check__label">{label}</span>
      <span className="ex-check__cell">
        <span className="ex-check__k">{t('ex.shell.rebuilt')}</span>
        <span className="num ex-check__v">{fx(rebuilt, 5)}</span>
      </span>
      <span className="ex-check__cell">
        <span className="ex-check__k">{t('ex.shell.returned')}</span>
        <span className="num ex-check__v">{fx(runtime, 5)}</span>
      </span>
      <span className="ex-check__cell">
        <span className="ex-check__k">|Δ|</span>
        <span className="num ex-check__v">
          {ok ? <CheckCircle size={14} weight="fill" aria-hidden="true" /> : <WarningCircle size={14} weight="fill" aria-hidden="true" />}
          {diff(d)}
          <span className="visually-hidden">{ok ? t('ex.shell.agree') : t('ex.shell.disagree')}</span>
        </span>
      </span>
    </div>
  )
}

/** A labelled term of a sum: "Σ positive terms  +12.3401". */
export function Term({ label, value, strong, signed = true }: { label: ReactNode; value: number; strong?: boolean; signed?: boolean }) {
  return (
    <div className={`ex-term${strong ? ' ex-term--strong' : ''}`}>
      <dt>{label}</dt>
      <dd className="num">{fx(value, 4, signed)}</dd>
    </div>
  )
}

export function Panel({ title, sub, children, className, id }: { title: ReactNode; sub?: ReactNode; children: ReactNode; className?: string; id?: string }) {
  return (
    <section className={`ex-panel ${className ?? ''}`} aria-labelledby={id}>
      <header className="ex-panel__head">
        <h4 className="ex-panel__title" id={id}>
          {title}
        </h4>
        {sub && <p className="ex-panel__sub">{sub}</p>}
      </header>
      {children}
    </section>
  )
}

export function Hint({ children }: { children: ReactNode }) {
  return (
    <p className="ex-hint">
      <Info size={14} aria-hidden="true" />
      <span>{children}</span>
    </p>
  )
}

// --------------------------------------------------------------------------- logits -> softmax -> probabilities
interface SoftmaxProps {
  /** logits returned by the graph */
  logits: ArrayLike<number>
  /** probabilities returned by the graph */
  probs: ArrayLike<number>
  classes: string[]
  /** false until the playback has reached the softmax */
  revealed: boolean
  selected?: number | null
  onSelect?: (k: number) => void
}

export function SoftmaxPanel({ logits, probs, classes, revealed, selected, onSelect }: SoftmaxProps) {
  const { t } = useI18n()
  const p = useMemo(() => softmax(logits), [logits])
  let m = -Infinity
  for (let i = 0; i < logits.length; i++) m = Math.max(m, logits[i])
  let s = 0
  const e = Array.from(logits, (z) => Math.exp(z - m))
  for (const v of e) s += v
  let worst = 0
  for (let i = 0; i < p.length; i++) worst = Math.max(worst, Math.abs(p[i] - probs[i]))
  let top = 0
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i
  return (
    <Panel id="ex-softmax-title" title={t('ex.shell.softmax.title')} className="ex-softmax">
      <p className="num ex-formula">
        p<sub>k</sub> = e<sup>z<sub>k</sub> − m</sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub> − m</sup>, m = max z = {fx(m, 4)}, Σ = {fx(s, 4)}
      </p>
      <div className="ex-softmax__table" role="table" aria-label={t('ex.shell.softmax.table')} data-revealed={revealed}>
        <div role="row" className="ex-softmax__row ex-softmax__row--head">
          <span role="columnheader">{t('ex.shell.softmax.class')}</span>
          <span role="columnheader" className="ex-nocase">z</span>
          <span role="columnheader" className="ex-nocase">e^(z−m)</span>
          <span role="columnheader" className="ex-nocase">p</span>
        </div>
        {classes.map((c, k) => (
          <div
            role="row"
            key={k}
            className="ex-softmax__row"
            data-top={k === top}
            aria-selected={selected === k}
            onClick={onSelect ? () => onSelect(k) : undefined}
            style={{ cursor: onSelect ? 'pointer' : undefined }}
          >
            <span role="cell" className="ex-softmax__class">
              {c}
              {k === top && <span className="ex-tag">{t('ex.shell.softmax.top')}</span>}
            </span>
            <span role="cell" className="num">
              {fx(logits[k], 3)}
            </span>
            <span role="cell" className="num">
              {fx(e[k], 4)}
            </span>
            <span role="cell" className="ex-softmax__p">
              <span className="ex-bar" aria-hidden="true">
                <span style={{ transform: `scaleX(${revealed ? Math.max(0, Math.min(1, probs[k])) : 0})` }} />
              </span>
              <span className="num">{(probs[k] * 100).toFixed(2)}%</span>
            </span>
          </div>
        ))}
      </div>
      <p className="ex-hint num">{t('ex.shell.softmax.check', { d: diff(worst) })}</p>
    </Panel>
  )
}

// --------------------------------------------------------------------------- image -> flatten -> standardise
interface PreprocessProps {
  pixels: Float32Array
  /** standardised input returned by the graph (784 values) */
  x: Float32Array
  mean: number
  std: number
  eps: number
  /** draw the 784-long strip (MLP), or keep the 28 x 28 layout (CNN reshapes it back) */
  flatten: boolean
  /** which of the three steps the playback has reached (0..3) */
  reached: number
  side: number
}

export function PreprocessStrip({ pixels, x, mean, std, eps, flatten, reached, side }: PreprocessProps) {
  const { t } = useI18n()
  const n = side * side
  const brightest = useMemo(() => {
    let b = 0
    for (let i = 1; i < n; i++) if (pixels[i] > pixels[b]) b = i
    return b
  }, [pixels, n])
  const [pix, setPix] = usePersisted<number | null>('pre.pixel', null)
  const i = pix ?? brightest
  const p = pixels[i]
  const rebuilt = (p - mean) / (std + eps)
  return (
    <div className={`ex-pre${flatten ? '' : ' ex-pre--noflat'}`}>
      <figure className="ex-pre__fig" data-on={reached >= 1}>
        <HeatMap values={pixels} width={side} height={side} mode="gray" max={1} label={t('ex.shell.pre.imageAria', { r: Math.floor(i / side), c: i % side })} selected={i} onSelect={setPix} />
        <figcaption>
          <b>{t('ex.shell.pre.image')}</b> <span className="num">{shape([side, side])}</span>
        </figcaption>
      </figure>
      {flatten && (
        <figure className="ex-pre__fig ex-pre__fig--strip" data-on={reached >= 2}>
          <FlatStrip values={pixels} n={n} sel={i} />
          <figcaption>
            <b>{t('ex.shell.pre.flatten')}</b> <span className="num">{n}</span>
            <span className="ex-pre__note">{t('ex.shell.pre.flattenNote', { side, n })}</span>
          </figcaption>
        </figure>
      )}
      <figure className="ex-pre__fig" data-on={reached >= 3}>
        <HeatMap values={x} width={side} height={side} mode="diverging" label={t('ex.shell.pre.xAria')} selected={i} onSelect={setPix} />
        <figcaption>
          <b>{t('ex.shell.pre.std')}</b> <span className="num">{flatten ? n : shape([side, side])}</span>
        </figcaption>
      </figure>
      <div className="ex-pre__calc">
        <p className="num ex-formula">
          x = (p/255 − {fx(mean, 4)}) / ({fx(std, 4)} + {eps.toExponential(0).replace('-', '−')})
        </p>
        <p className="ex-pre__px">{t('ex.shell.pre.pixel', { r: Math.floor(i / side), c: i % side, i })}</p>
        <dl className="ex-terms">
          <Term label="p" value={Math.round(p * 255)} signed={false} />
          <Term label="p/255" value={p} signed={false} />
        </dl>
        <CheckRow label="x" rebuilt={rebuilt} runtime={x[i]} />
        <Hint>{t('ex.shell.pre.hint')}</Hint>
      </div>
    </div>
  )
}

/** The picture unrolled row after row into one line of n values (canvas, one column per value). */
function FlatStrip({ values, n, sel }: { values: Float32Array; n: number; sel: number }) {
  const rows = 2
  const per = Math.ceil(n / rows)
  return (
    <div className="ex-strip" role="img" aria-label={shape([n])}>
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="ex-strip__row">
          <HeatMap
            values={values.subarray(r * per, Math.min(n, (r + 1) * per))}
            width={Math.min(per, n - r * per)}
            height={1}
            mode="gray"
            max={1}
            label={`${r * per}–${Math.min(n, (r + 1) * per) - 1}`}
            selected={sel >= r * per && sel < (r + 1) * per ? sel - r * per : null}
            className="ex-strip__heat"
            style={{ aspectRatio: 'auto', height: 18 }}
          />
          <span className="num ex-strip__idx">
            {r * per}–{Math.min(n, (r + 1) * per) - 1}
          </span>
        </div>
      ))}
    </div>
  )
}
