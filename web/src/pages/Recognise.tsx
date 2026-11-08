import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { DigitPad, cropOf, type CropRect, type DigitPadHandle, type PadChange } from '../components/DigitPad'
import { Histogram } from '../components/Histogram'
import { Inside } from '../components/inside/Inside'
import { Mnist28 } from '../components/Mnist28'
import { useI18n } from '../i18n'
import { SAMPLE_DIGITS } from '../lib/samples'
import { useModelStatuses, useModelStore, useSiteData } from '../lib/context'
import { bytes, int, latency, pct } from '../lib/format'
import { MNIST_SIZE, preprocessMnist, type Preprocessed } from '../lib/preprocess'
import type { ModelId, Prediction } from '../lib/types'

type Phase = 'blank' | 'empty' | 'ready'

/** The three small models load as soon as someone reaches for the pen. AdaBoost is 16 MB, so it waits for a click. */
const EAGER: ModelId[] = ['naive_bayes', 'softmax', 'mlp']

interface Verdict {
  digit: number
  votes: number
  of: number
}

/** Majority vote. A tie is broken by the mean probability the tied digits received. */
function consensusOf(preds: Partial<Record<ModelId, Prediction>>): Verdict | null {
  const list = Object.values(preds) as Prediction[]
  if (list.length === 0) return null
  const tally = new Map<number, { n: number; p: number }>()
  for (const p of list) {
    const cur = tally.get(p.top) ?? { n: 0, p: 0 }
    tally.set(p.top, { n: cur.n + 1, p: cur.p + p.probs[p.top] })
  }
  let best: [number, { n: number; p: number }] | null = null
  for (const e of tally) if (!best || e[1].n > best[1].n || (e[1].n === best[1].n && e[1].p > best[1].p)) best = e
  return best ? { digit: best[0], votes: best[1].n, of: list.length } : null
}

export default function Recognise() {
  const { t } = useI18n()
  const { manifest } = useSiteData()
  const store = useModelStore()
  const statuses = useModelStatuses(store)
  const ids = store.ids()
  const [live, setLive] = useState(true)

  const padRef = useRef<DigitPadHandle>(null)
  const [pre, setPre] = useState<Preprocessed | null>(null)
  const [crop, setCrop] = useState<CropRect | null>(null)
  const [phase, setPhase] = useState<Phase>('blank')
  const [version, setVersion] = useState(0)
  const [manualVersion, setManualVersion] = useState(-1)
  const [preds, setPreds] = useState<Partial<Record<ModelId, Prediction>>>({})
  const [predVersion, setPredVersion] = useState(-1)
  const [announce, setAnnounce] = useState('')
  const announceTimer = useRef(0)

  const loadEager = useCallback(() => {
    for (const id of EAGER) {
      const s = store.getSnapshot()[id]
      if (ids.includes(id) && (!s || s.state === 'idle')) void store.load(id)
    }
  }, [store, ids])

  const onPadChange = useCallback((c: PadChange) => {
    if (!c.image) {
      setPre(null)
      setCrop(null)
      setPhase('blank')
      setVersion((v) => v + 1)
      return
    }
    const out = preprocessMnist(c.image)
    setPre(out)
    setCrop(out && c.kind === 'draw' ? cropOf(out.bbox, c.image.width, c.image.height) : null)
    setPhase(out ? 'ready' : 'empty')
    setVersion((v) => v + 1)
  }, [])

  const readyKey = ids.filter((id) => statuses[id]?.state === 'ready').join(',')
  const armed = live || manualVersion === version

  useEffect(() => {
    if (!pre) {
      setPreds({})
      setPredVersion(-1)
      return
    }
    if (!armed || !readyKey) return
    let cancelled = false
    const v = version
    void (async () => {
      const next: Partial<Record<ModelId, Prediction>> = {}
      for (const id of readyKey.split(',') as ModelId[]) {
        try {
          next[id] = await store.predict(id, pre.pixels)
        } catch {
          /* a failed single prediction leaves that model without a value */
        }
        if (cancelled) return
      }
      setPreds(next)
      setPredVersion(v)
    })()
    return () => {
      cancelled = true
    }
  }, [pre, armed, readyKey, version, store])

  const stale = !!pre && Object.keys(preds).length > 0 && predVersion !== version
  const verdict = useMemo(() => consensusOf(preds), [preds])
  const nameOf = (id: ModelId) => t(`model.${id}`)

  const verdictText = !verdict
    ? null
    : verdict.of === 1
      ? t('rec.consensus.single', { name: nameOf(Object.keys(preds)[0] as ModelId) })
      : verdict.votes === verdict.of
        ? t('rec.compare.agree', { n: verdict.of, digit: manifest.classes[verdict.digit] })
        : t('rec.consensus.majority', { k: verdict.votes, n: verdict.of, digit: manifest.classes[verdict.digit] })

  // Screen-reader announcement, throttled so that live drawing does not chatter.
  useEffect(() => {
    window.clearTimeout(announceTimer.current)
    announceTimer.current = window.setTimeout(() => {
      if (verdict && verdictText && !stale) setAnnounce(`${t('rec.consensus.title')}: ${manifest.classes[verdict.digit]}. ${verdictText}`)
      else if (phase === 'blank') setAnnounce('')
    }, 700)
    return () => window.clearTimeout(announceTimer.current)
  }, [verdict, verdictText, stale, phase, manifest.classes, t])

  const ds = manifest.dataset
  const inputSize = `${MNIST_SIZE} × ${MNIST_SIZE}`

  return (
    <div className="console">
      <section className="intro">
        <h1 className="display">{t('rec.title')}</h1>
        <p className="lede">{t('rec.lede')}</p>
        <dl className="form-facts">
          <div>
            <dt>{t('rec.form.dataset')}</dt>
            <dd>MNIST</dd>
          </div>
          {ds?.n_train != null && (
            <div>
              <dt>{t('rec.form.train')}</dt>
              <dd className="num">{int(ds.n_train)}</dd>
            </div>
          )}
          {ds?.n_test != null && (
            <div>
              <dt>{t('rec.form.test')}</dt>
              <dd className="num">{int(ds.n_test)}</dd>
            </div>
          )}
          <div>
            <dt>{t('rec.form.input')}</dt>
            <dd className="num">{inputSize}</dd>
          </div>
          <div>
            <dt>{t('rec.form.classes')}</dt>
            <dd className="num">{manifest.classes.length}</dd>
          </div>
        </dl>
      </section>

      <div className="deck">
        <section className="screen" aria-labelledby="in-h">
          <header className="panel-head">
            <h2 id="in-h" className="panel-head__title">
              <span className="panel-head__no" aria-hidden="true">
                01
              </span>
              {t('rec.step.draw')}
            </h2>
          </header>
          <DigitPad ref={padRef} crop={crop} onChange={onPadChange} onIntent={loadEager} />
          <div className="screen__actions">
            <button
              type="button"
              className="btn btn--primary"
              onClick={() => {
                loadEager()
                setManualVersion(version)
              }}
              disabled={phase !== 'ready'}
            >
              {t('rec.predict')}
            </button>
            <label className="check">
              <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
              <span>{t('rec.live')}</span>
            </label>
          </div>
          <div className="samples">
            <p className="samples__label">{t('rec.samples.label')}</p>
            <ul className="samples__row">
              {SAMPLE_DIGITS.map((d) => (
                <li key={d}>
                  <button type="button" className="chip" aria-label={t('rec.samples.draw', { digit: d })} onClick={() => padRef.current?.drawSample(d)}>
                    <span className="num" aria-hidden="true">
                      {d}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section className="readout" aria-labelledby="out-h">
          <header className="panel-head">
            <h2 id="out-h" className="panel-head__title">
              <span className="panel-head__no" aria-hidden="true">
                02
              </span>
              {t('rec.step.answer')}
            </h2>
          </header>

          <div className={`verdict${verdict && !stale ? ' has-result' : ''}${stale ? ' is-stale' : ''}`}>
            <div className="verdict__eye">
              <Mnist28 pixels={pre?.pixels ?? null} />
              <span className="verdict__eyelabel">{t('rec.step.see')}</span>
            </div>
            <div className="verdict__main">
              <p className="verdict__title">{t('rec.consensus.title')}</p>
              <p className={`verdict__digit${verdict ? '' : ' is-none'}`} key={verdict?.digit ?? 'none'} aria-hidden="true">
                {verdict ? manifest.classes[verdict.digit] : '–'}
              </p>
            </div>
            <div className="verdict__text">
              <p className="verdict__say">
                {verdictText ??
                  (phase === 'empty' ? t('rec.nothing.title') : phase === 'ready' ? t('rec.consensus.waiting') : t('rec.blank.title'))}
              </p>
              <p className="caption">
                {stale
                  ? t('rec.stale')
                  : phase === 'empty'
                    ? t('rec.nothing.text')
                    : phase === 'blank'
                      ? t('rec.blank.text')
                      : pre
                        ? t('rec.prep.done', { x: pre.com.x.toFixed(1), y: pre.com.y.toFixed(1) })
                        : ''}
              </p>
            </div>
          </div>

          <ul className="models">
            {ids.map((id) => {
              const s = statuses[id]
              const p = preds[id]
              const size = bytes(manifest.models[id]?.size_bytes)
              const dissent = !!(p && verdict && verdict.of > 1 && p.top !== verdict.digit)
              const entry = manifest.models[id]
              const v = entry?.validation
              const state = s?.state ?? 'idle'
              return (
                <li key={id} className={`mcard${dissent ? ' is-dissent' : ''}${p ? ' has-pred' : ''}`}>
                  <div className="mcard__head">
                    <h3 className="mcard__name">{nameOf(id)}</h3>
                    <span className="mcard__state">
                      <span className={`dot dot--${state}`} aria-hidden="true" />
                      {state === 'ready' ? size : state === 'loading' ? t('rec.compare.loading') : state === 'error' ? t('rec.compare.failed') : size}
                    </span>
                  </div>

                  <div className="mcard__read">
                    <p className="mcard__digit num" aria-hidden="true">
                      {p && pre ? manifest.classes[p.top] : '–'}
                    </p>
                    <div className="mcard__meta">
                      {p && pre ? (
                        <>
                          <p className="mcard__conf num">{pct(p.probs[p.top], 0)}</p>
                          <p className="mcard__label">
                            {id === 'adaboost' ? t('out.label.adaboost') : t('out.short.confidence')}
                            {id === 'adaboost' ? ` · ${t('out.notCalibrated')}` : ''}
                          </p>
                          <p className="mcard__label num">
                            {latency(p.ms)}
                            {dissent && (
                              <>
                                {' · '}
                                <span className="mcard__tag">{t('rec.card.dissent')}</span>
                              </>
                            )}
                          </p>
                        </>
                      ) : state === 'idle' && id === 'adaboost' ? (
                        <button type="button" className="btn btn--quiet btn--small" onClick={() => void store.load(id)}>
                          {t('rec.card.load', { size })}
                        </button>
                      ) : state === 'error' ? (
                        <button type="button" className="btn btn--quiet btn--small" onClick={() => void store.load(id)}>
                          {t('common.retry')}
                        </button>
                      ) : (
                        <p className="mcard__label">
                          {state === 'loading'
                            ? t('rec.compare.loading')
                            : state === 'ready'
                              ? pre
                                ? t('rec.compare.waitPredict')
                                : t('rec.compare.waitDrawing')
                              : t('rec.compare.notLoaded')}
                        </p>
                      )}
                    </div>
                  </div>

                  <Histogram probs={p && pre ? p.probs : null} classes={manifest.classes} stale={stale} />

                  <details className="mcard__about">
                    <summary>{t('rec.card.about')}</summary>
                    <p className="note">{t(`out.note.${id}`)}</p>
                    {v?.n_checked ? (
                      <p className="note note--muted">
                        {t('out.checked', {
                          n: int(v.n_checked),
                          diff: v.max_abs_prob_diff == null ? t('common.na') : v.max_abs_prob_diff === 0 ? '0' : v.max_abs_prob_diff.toExponential(1),
                          agree: pct(v.argmax_agreement, 1),
                        })}
                      </p>
                    ) : null}
                  </details>
                </li>
              )
            })}
          </ul>
          <p className="visually-hidden" role="status" aria-live="polite">
            {announce}
          </p>
        </section>
      </div>

      <Inside pixels={pre?.pixels ?? null} />
    </div>
  )
}
