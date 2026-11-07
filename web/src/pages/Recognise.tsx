import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { DigitPad, cropOf, type CropRect, type DigitPadHandle, type PadChange } from '../components/DigitPad'
import { Mnist28 } from '../components/Mnist28'
import { ModelPicker } from '../components/ModelPicker'
import { ProbBars } from '../components/ProbBars'
import { Segmented } from '../components/Segmented'
import { useI18n } from '../i18n'
import { SAMPLE_DIGITS } from '../lib/samples'
import { useModelStatuses, useModelStore, useSiteData } from '../lib/context'
import { bytes, int, latency, pct } from '../lib/format'
import { MNIST_SIZE, preprocessMnist, type Preprocessed } from '../lib/preprocess'
import type { ModelId, Prediction } from '../lib/types'

type Phase = 'blank' | 'empty' | 'ready'

const DEFAULT_MODEL: ModelId = 'mlp'

export default function Recognise() {
  const { t } = useI18n()
  const { manifest } = useSiteData()
  const store = useModelStore()
  const statuses = useModelStatuses(store)
  const ids = store.ids()
  const [selected, setSelected] = useState<ModelId>(ids.includes(DEFAULT_MODEL) ? DEFAULT_MODEL : ids[0])
  const [mode, setMode] = useState<'top3' | 'all'>('top3')
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

  const ensureLoaded = useCallback(
    (id: ModelId) => {
      const s = store.getSnapshot()[id]
      if (!s || s.state === 'idle') void store.load(id)
    },
    [store],
  )

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

  const select = (id: ModelId) => {
    setSelected(id)
    const s = store.getSnapshot()[id]
    if (!s || s.state === 'idle' || s.state === 'error') void store.load(id)
  }

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

  const status = statuses[selected]
  const pred = preds[selected]
  const stale = !!pre && !!pred && predVersion !== version
  const top = pred ? manifest.classes[pred.top] : null
  const nameOf = (id: ModelId) => t(`model.${id}`)
  const labelOf = (id: ModelId) => t(`out.label.${id}`)

  // Screen-reader announcement, throttled so that live drawing does not chatter.
  useEffect(() => {
    window.clearTimeout(announceTimer.current)
    announceTimer.current = window.setTimeout(() => {
      if (pred && !stale) {
        setAnnounce(
          t(selected === 'adaboost' ? 'rec.announceUncal' : 'rec.announce', {
            name: nameOf(selected),
            digit: top ?? '',
            value: pct(pred.probs[pred.top], 0),
            label: labelOf(selected),
          }),
        )
      } else if (phase === 'blank') setAnnounce('')
    }, 700)
    return () => window.clearTimeout(announceTimer.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- nameOf and labelOf only depend on t
  }, [pred, stale, selected, top, phase, t])

  const agreement = useMemo(() => {
    const entries = Object.entries(preds) as [ModelId, Prediction][]
    if (entries.length < 2) return null
    const tops = new Set(entries.map(([, p]) => p.top))
    return tops.size === 1 ? t('rec.compare.agree', { n: entries.length, digit: manifest.classes[entries[0][1].top] }) : t('rec.compare.disagree')
  }, [preds, manifest.classes, t])

  const notLoaded = ids.filter((id) => !statuses[id] || statuses[id]!.state === 'idle')
  const entry = manifest.models[selected]
  const v = entry?.validation
  const ds = manifest.dataset
  const inputSize = `${MNIST_SIZE} × ${MNIST_SIZE}`

  return (
    <div className="recognise">
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

      <div className="slip">
        <div className="slip__main">
          <section className="step" aria-labelledby="in-h">
            <h2 id="in-h" className="step__label">
              <span className="step__no num">1</span>
              {t('rec.step.draw')}
            </h2>
            <DigitPad ref={padRef} crop={crop} onChange={onPadChange} onIntent={() => ensureLoaded(selected)} />
            <div className="slip__actions">
              <button
                type="button"
                className="btn btn--primary"
                onClick={() => {
                  ensureLoaded(selected)
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
        </div>

        <div className="slip__perf" aria-hidden="true" />

        <div className="slip__stub">
          <section className="step step--prep" aria-labelledby="prep-h">
            <h2 id="prep-h" className="step__label">
              <span className="step__no num">2</span>
              {t('rec.step.see')}
            </h2>
            <div className="prep">
              <Mnist28 pixels={pre?.pixels ?? null} />
              <p className="caption">{pre ? t('rec.prep.done', { x: pre.com.x.toFixed(1), y: pre.com.y.toFixed(1) }) : t('rec.prep.empty')}</p>
            </div>
          </section>

          <section className="step" aria-labelledby="out-h">
            <h2 id="out-h" className="step__label">
              <span className="step__no num">3</span>
              {t('rec.step.answer')}
            </h2>
            <ModelPicker ids={ids} selected={selected} statuses={statuses} manifest={manifest} onSelect={select} />

            <div className="result" aria-busy={status?.state === 'loading'}>
              {status?.state === 'error' ? (
                <div className="result__error" role="alert">
                  <p className="msg msg--error">{t(status.key, status.params)}</p>
                  <button type="button" className="btn btn--primary" onClick={() => void store.load(selected)}>
                    {t('common.retry')}
                  </button>
                </div>
              ) : phase === 'blank' ? (
                <div className="result__empty">
                  <p className="result__title">{t('rec.blank.title')}</p>
                  <p className="caption">{t('rec.blank.text')}</p>
                </div>
              ) : phase === 'empty' ? (
                <div className="result__empty">
                  <p className="result__title">{t('rec.nothing.title')}</p>
                  <p className="caption">{t('rec.nothing.text')}</p>
                </div>
              ) : !pred && status?.state !== 'ready' ? (
                <div className="result__loading" aria-label={t('picker.loadingAria')}>
                  <div className="skeleton skeleton--digit" />
                  <div className="skeleton skeleton--line" />
                  <div className="skeleton skeleton--line" />
                  <p className="caption">{t('rec.loading.text', { name: nameOf(selected), size: bytes(entry?.size_bytes) })}</p>
                </div>
              ) : !pred ? (
                <div className="result__empty">
                  <p className="result__title">{t('rec.ready.title')}</p>
                  <p className="caption">{t('rec.ready.text')}</p>
                </div>
              ) : (
                <div className={`result__body${stale ? ' is-stale' : ''}`}>
                  <div className="result__head">
                    <div className="result__digit" key={top} aria-hidden="true">
                      {top}
                    </div>
                    <div>
                      <p className="result__conf">
                        <span className="num">{pct(pred.probs[pred.top], 1)}</span> {labelOf(selected)}
                      </p>
                      {selected === 'adaboost' && <p className="result__qual">{t('out.notCalibrated')}</p>}
                      <p className="caption">{t('rec.runTime', { name: nameOf(selected), time: latency(pred.ms) })}</p>
                      {pred.probs[pred.top] < 0.5 && <p className="caption">{t('rec.lowConfidence')}</p>}
                      {stale && <p className="caption">{t('rec.stale')}</p>}
                    </div>
                  </div>
                  <div className="result__bars">
                    <Segmented
                      legend={t('rec.bars.label')}
                      hideLegend
                      value={mode}
                      onChange={setMode}
                      options={[
                        { value: 'top3', label: t('rec.bars.top3') },
                        { value: 'all', label: t('rec.bars.all') },
                      ]}
                    />
                    <ProbBars probs={pred.probs} classes={manifest.classes} mode={mode} stale={stale} />
                  </div>
                </div>
              )}
            </div>

            <p className="note">{t(`out.note.${selected}`)}</p>
            {v?.n_checked ? (
              <p className="note note--muted">
                {t('out.checked', {
                  n: int(v.n_checked),
                  diff: v.max_abs_prob_diff == null ? t('common.na') : v.max_abs_prob_diff === 0 ? '0' : v.max_abs_prob_diff.toExponential(1),
                  agree: pct(v.argmax_agreement, 1),
                })}
              </p>
            ) : null}
            <p className="visually-hidden" role="status" aria-live="polite">
              {announce}
            </p>
          </section>
        </div>
      </div>

      <section className="compare" aria-labelledby="cmp-h">
        <div className="compare__head">
          <h2 id="cmp-h" className="h3">
            {t('rec.compare.title')}
          </h2>
          {notLoaded.length > 0 && (
            <button type="button" className="btn btn--quiet" onClick={() => notLoaded.forEach((id) => void store.load(id))}>
              {t('rec.compare.loadOthers')}
            </button>
          )}
        </div>
        <ul className="compare__row">
          {ids.map((id) => {
            const s = statuses[id]
            const p = preds[id]
            return (
              <li key={id} className={`compare__cell${id === selected ? ' is-selected' : ''}`}>
                <p className="compare__name">{nameOf(id)}</p>
                {p && pre ? (
                  <>
                    <p className="compare__digit num">{manifest.classes[p.top]}</p>
                    <p className="caption">
                      <span className="num">
                        {id === 'adaboost'
                          ? t('rec.compare.voteShare', { value: pct(p.probs[p.top], 0) })
                          : t('rec.compare.confidence', { value: pct(p.probs[p.top], 0) })}
                      </span>
                      {id === 'adaboost' ? ` (${t('out.notCalibrated')})` : ''} · <span className="num">{latency(p.ms)}</span>
                    </p>
                  </>
                ) : (
                  <>
                    <p className="compare__digit compare__digit--none" aria-hidden="true">
                      {'\u2013'}
                    </p>
                    <p className="caption">
                      {s?.state === 'error'
                        ? t('rec.compare.failed')
                        : s?.state === 'loading'
                          ? t('rec.compare.loading')
                          : s?.state === 'ready'
                            ? pre
                              ? t('rec.compare.waitPredict')
                              : t('rec.compare.waitDrawing')
                            : t('rec.compare.notLoaded')}
                    </p>
                  </>
                )}
              </li>
            )
          })}
        </ul>
        {agreement && <p className="caption">{agreement}</p>}
      </section>
    </div>
  )
}
