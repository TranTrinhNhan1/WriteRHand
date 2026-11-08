import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { Check, MagnifyingGlass, Warning } from '@phosphor-icons/react'
import { ClassBars } from '../components/ClassBars'
import { DigitPad, cropOf, type CropRect, type DigitPadHandle, type PadChange } from '../components/DigitPad'
import { Mnist28 } from '../components/Mnist28'
import { Thumb28 } from '../components/Thumb28'
import { MODEL_ORDER, augId, type DatasetEntry, type ModelId, type ModelKind } from '../engine/contract'
import { useBundle, useExecution, type ExecutionState } from '../engine/react'
import type { ModelStatus } from '../engine/store'
import { useI18n } from '../i18n'
import { useChoice } from '../lib/choice'
import { bytes, int, latency, pct } from '../lib/format'
import { MNIST_SIZE, preprocessMnist, type Preprocessed } from '../lib/preprocess'
import { SAMPLE_DIGITS } from '../lib/samples'
import { consensusOf, datasetIdsOf, isDigitDataset, modelChoices, modelPairs, variantId, type Answer, type ModelChoice } from '../lib/selection'
import { useTestImages, type TestImage } from '../lib/testImages'

const Explorer = lazy(() => import('../components/explorer/Explorer'))

/** Where the 784 pixels the models receive come from. */
type Source =
  | { kind: 'blank' }
  /** the pad has something, but the pipeline found no ink */
  | { kind: 'empty' }
  | { kind: 'pad'; pre: Preprocessed; crop: CropRect | null; upload: boolean }
  | { kind: 'test'; dataset: string; image: TestImage }

const pixelsOf = (s: Source): Float32Array | null => (s.kind === 'pad' ? s.pre.pixels : s.kind === 'test' ? s.image.pixels : null)

/** Every model id a dataset can have: each architecture and its augmented variant (a constant list). */
const ALL_IDS: ModelId[] = MODEL_ORDER.flatMap((k) => [k, augId(k)])

/**
 * Runs every model of the dataset on the same pixels. `ALL_IDS` is a constant, so the hooks are
 * always called in the same order; a model the dataset does not offer (or that failed its export
 * check, or is not loaded yet) gets no input and stays idle.
 */
function useAllExecutions(dataset: string, choices: ModelChoice[], pixels: Float32Array | null): Record<ModelId, ExecutionState> {
  const out = {} as Record<ModelId, ExecutionState>
  for (const id of ALL_IDS) {
    const c = choices.find((x) => x.id === id)
    // eslint-disable-next-line react-hooks/rules-of-hooks -- fixed call order (constant list)
    out[id] = useExecution(dataset, id, c?.selectable ? pixels : null, !!c?.autoLoad)
  }
  return out
}

function kindLabel(t: ReturnType<typeof useI18n>['t'], ds: DatasetEntry): string {
  if (ds.kind === 'digits') return t('ds.kind.digits')
  if (ds.kind === 'clothing') return t('ds.kind.clothing')
  return ds.kind
}

function DatasetPicker() {
  const { t } = useI18n()
  const { manifest } = useBundle()
  const { dataset, setDataset } = useChoice()
  const ids = datasetIdsOf(manifest)
  return (
    <fieldset className="dspick">
      <legend className="dspick__legend">
        <span className="panel-head__no" aria-hidden="true">
          01
        </span>
        {t('rec.ds.legend')}
      </legend>
      <div className="dspick__row">
        {ids.map((id) => {
          const ds = manifest.datasets[id]
          const n = Object.keys(ds.models).length
          return (
            <label key={id} className="dspick__opt">
              <input type="radio" name="dataset" value={id} checked={id === dataset} onChange={() => setDataset(id)} />
              <span className="dspick__card">
                <span className="dspick__top">
                  <span className="dspick__title">{ds.title}</span>
                  <span className="dspick__tick" aria-hidden="true">
                    <Check size={14} weight="bold" />
                  </span>
                </span>
                <span className="dspick__kind">{t('rec.ds.kindLine', { kind: kindLabel(t, ds), n: ds.classes.length })}</span>
                <span className="dspick__meta num">{t('rec.ds.models', { n })}</span>
              </span>
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}

function ReadingAs({ ds }: { ds: DatasetEntry }) {
  const { t } = useI18n()
  const d = ds.dataset
  return (
    <div className="reading" aria-live="polite">
      <p className="reading__line">
        <span className="reading__key">{t('rec.ds.readingAs')}</span>
        <strong className="reading__ds">{ds.title}</strong>
        <span className="reading__sep" aria-hidden="true">
          ·
        </span>
        <span className="reading__key">{t('rec.ds.classes', { n: ds.classes.length })}</span>
      </p>
      <ol className="reading__classes" aria-label={t('rec.ds.classList', { title: ds.title })}>
        {ds.classes.map((c, i) => (
          <li key={`${i}-${c}`}>
            {c !== String(i) && (
              <span className="reading__idx num" aria-hidden="true">
                {i}
              </span>
            )}
            {c}
          </li>
        ))}
      </ol>
      <dl className="form-facts">
        <div>
          <dt>{t('rec.form.dataset')}</dt>
          <dd>{ds.title}</dd>
        </div>
        <div>
          <dt>{t('rec.form.kind')}</dt>
          <dd>{kindLabel(t, ds)}</dd>
        </div>
        <div>
          <dt>{t('rec.form.train')}</dt>
          <dd className="num">{int(d?.n_train)}</dd>
        </div>
        <div>
          <dt>{t('rec.form.test')}</dt>
          <dd className="num">{int(d?.n_test)}</dd>
        </div>
        <div>
          <dt>{t('rec.form.input')}</dt>
          <dd className="num">{`${MNIST_SIZE} × ${MNIST_SIZE}`}</dd>
        </div>
      </dl>
    </div>
  )
}

function TestImages({ dataset, ds, chosen, onPick, onIntent }: { dataset: string; ds: DatasetEntry; chosen: TestImage | null; onPick: (img: TestImage) => void; onIntent: () => void }) {
  const { t } = useI18n()
  const { manifest, baseUrl } = useBundle()
  const [attempt, setAttempt] = useState(0)
  const state = useTestImages(manifest, baseUrl, dataset, true, attempt)
  const groups = useMemo(() => {
    if (state.state !== 'ready') return []
    return ds.classes.map((name, k) => ({ name, k, images: state.images.filter((im) => im.label === k) })).filter((g) => g.images.length > 0)
  }, [state, ds.classes])
  return (
    <div className="tests">
      <p className="samples__label">{t('rec.test.label', { title: ds.title })}</p>
      {state.state === 'error' ? (
        <div className="tests__msg">
          <p className="msg msg--error">{t(state.error.key, state.error.params)}</p>
          <button type="button" className="btn btn--quiet btn--small" onClick={() => setAttempt((a) => a + 1)}>
            {t('common.retry')}
          </button>
        </div>
      ) : state.state !== 'ready' ? (
        <div className="tests__skel" aria-busy="true" aria-label={t('rec.test.loading')}>
          {Array.from({ length: 6 }, (_, i) => (
            <span key={i} className="skeleton" />
          ))}
        </div>
      ) : (
        <ul className="tests__groups">
          {groups.map((g) => (
            <li key={g.k} className="tests__group">
              <span className="tests__name" title={g.name}>
                {g.name}
              </span>
              <span className="tests__row">
                {g.images.map((im) => {
                  const on = chosen?.index === im.index && chosen.label === im.label
                  return (
                    <button
                      key={im.index}
                      type="button"
                      className="tests__btn"
                      aria-pressed={on}
                      aria-label={t('rec.test.aria', { index: im.index, name: g.name })}
                      title={t('rec.test.aria', { index: im.index, name: g.name })}
                      onPointerEnter={onIntent}
                      onFocus={onIntent}
                      onClick={() => onPick(im)}
                    >
                      <Thumb28 pixels={im.pixels} />
                    </button>
                  )
                })}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

interface CardProps {
  choice: ModelChoice
  dataset: string
  classes: string[]
  exec: ExecutionState
  status: ModelStatus
  hasInput: boolean
  dissent: boolean
  selected: boolean
  onSelect: () => void
  onLoad: () => void
  /** shown in a plain / augmented pair: the row carries the model name, the card its variant */
  paired?: boolean
}

function ModelCard({ choice, classes, exec, status, hasInput, dissent, selected, onSelect, onLoad, paired }: CardProps) {
  const { t } = useI18n()
  const { kind, entry, selectable } = choice
  const out = entry.trace.output
  const isVote = out.kind === 'vote_share'
  const ready = exec.state === 'ready' ? exec.execution : null
  const size = bytes(choice.bytes)
  const top = ready ? classes[ready.top] : null
  const v = entry.validation
  const st = !selectable ? 'off' : status.state
  const quantity = isVote ? t('out.short.voteShare') : t('out.short.probability')

  let line: React.ReactNode
  if (!selectable) {
    line = <p className="mcard__label">{t('rec.card.blocked')}</p>
  } else if (ready) {
    line = (
      <>
        <p className="mcard__conf num">{pct(ready.probs[ready.top], 0)}</p>
        <p className="mcard__label">
          {quantity}
          {!out.calibrated ? ` · ${t('out.notCalibrated')}` : ''}
        </p>
        <p className="mcard__label num">
          {latency(ready.ms)}
          {dissent && (
            <>
              {' · '}
              <span className="mcard__tag">{t('rec.card.dissent')}</span>
            </>
          )}
        </p>
      </>
    )
  } else if (status.state === 'idle' && !choice.autoLoad) {
    line = (
      <button type="button" className="btn btn--quiet btn--small" onClick={onLoad}>
        {t('rec.card.load', { size })}
      </button>
    )
  } else if (status.state === 'error') {
    line = (
      <>
        <p className="mcard__label mcard__label--err">{t(status.key, status.params)}</p>
        <button type="button" className="btn btn--quiet btn--small" onClick={onLoad}>
          {t('common.retry')}
        </button>
      </>
    )
  } else {
    line = (
      <p className="mcard__label">
        {status.state === 'loading'
          ? t('rec.compare.loading')
          : status.state === 'ready'
            ? hasInput
              ? t('rec.compare.running')
              : t('rec.compare.waitDrawing')
            : t('rec.compare.notLoaded')}
      </p>
    )
  }

  return (
    <li
      className={`mcard${dissent ? ' is-dissent' : ''}${ready ? ' has-pred' : ''}${selected ? ' is-selected' : ''}${selectable ? '' : ' is-off'}`}
      data-variant={choice.augmented ? 'augmented' : 'plain'}
    >
      <div className="mcard__head">
        <h3 className="mcard__name">
          {/* in a pair the row shows the model name and the column header the variant */}
          <span className={paired ? 'visually-hidden' : undefined}>
            {t(`model.${kind}`)}
            {paired ? ` ${choice.augmented ? t('rec.pair.augmented') : t('rec.pair.plain')}` : ''}
          </span>
        </h3>
        <span className="mcard__state">
          <span className={`dot dot--${st}`} aria-hidden="true" />
          {st === 'loading' ? t('rec.compare.loading') : st === 'error' ? t('rec.compare.failed') : size}
        </span>
      </div>

      <div className="mcard__read">
        <p className={`mcard__digit num${top && top.length > 2 ? ' is-word' : ''}`} aria-hidden={!top}>
          {top ?? '–'}
        </p>
        <div className="mcard__meta">{line}</div>
      </div>

      <ClassBars values={ready ? ready.probs : null} classes={classes} quantity={quantity} dissent={dissent} />

      <div className="mcard__foot">
        <button type="button" className="mcard__inspect" aria-pressed={selected} disabled={!selectable} onClick={onSelect}>
          <MagnifyingGlass size={14} weight="bold" aria-hidden="true" />
          {selected ? t('rec.card.inspecting') : t('rec.card.inspect')}
        </button>
        <details className="mcard__about">
          <summary>{t('rec.card.about')}</summary>
          <p className="note">{t(`out.note.${kind}`)}</p>
          {v?.n_checked ? (
            <p className="note note--muted">
              {t('out.checked', {
                n: int(v.n_checked),
                diff: v.max_abs_prob_diff === 0 ? '0' : v.max_abs_prob_diff.toExponential(1),
                agree: pct(v.argmax_agreement, 1),
              })}
            </p>
          ) : null}
        </details>
      </div>
    </li>
  )
}

function ModelPicker({ ds, choices, value, onChange }: { ds: DatasetEntry; choices: ModelChoice[]; value: ModelId | null; onChange: (id: ModelId) => void }) {
  const { t } = useI18n()
  const current = choices.find((c) => c.id === value) ?? null
  const plain = choices.filter((c) => !c.augmented)
  const aug = !!current?.augmented
  const pick = (kind: ModelKind) => onChange(variantId(ds, kind, aug) ?? kind)
  const hasVariant = !!current && !!variantId(ds, current.kind, !aug)
  return (
    <div className="mpick">
      <fieldset className="mpick__set">
        <legend className="visually-hidden">{t('rec.explore.pick')}</legend>
        <div className="mpick__row">
          {plain.map((c) => {
            const shown = (current?.kind === c.kind ? current : null) ?? choices.find((x) => x.id === variantId(ds, c.kind, aug)) ?? c
            return (
              <label key={c.kind} className={`mpick__opt${shown.selectable ? '' : ' is-off'}`}>
                <input type="radio" name="explore-model" value={c.kind} checked={current?.kind === c.kind} disabled={!shown.selectable} onChange={() => pick(c.kind)} />
                <span className="mpick__pill">
                  <span className="mpick__name">{t(`model.${c.kind}`)}</span>
                  <span className="mpick__meta num">
                    {shown.selectable ? (shown.autoLoad ? bytes(shown.bytes) : t('rec.explore.download', { size: bytes(shown.bytes) })) : t('rec.explore.blocked')}
                  </span>
                </span>
              </label>
            )
          })}
        </div>
      </fieldset>
      {(hasVariant || aug) && current && (
        <div className="mpick__variant" role="group" aria-label={t('rec.pair.variant')}>
          {[false, true].map((a) => {
            const id = variantId(ds, current.kind, a)
            return (
              <button key={String(a)} type="button" className="mpick__vbtn" aria-pressed={aug === a} disabled={!id} onClick={() => id && onChange(id)}>
                {a ? t('rec.pair.augmented') : t('rec.pair.plain')}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

/** Downloads every augmented variant of the dataset that is not loaded yet. */
function LoadVariants({ choices, dataset, onLoad }: { choices: ModelChoice[]; dataset: string; onLoad: (ids: ModelId[]) => void }) {
  const { t } = useI18n()
  const { registry } = useBundle()
  const statuses = useSyncExternalStore(registry.subscribe, registry.getSnapshot)
  const waiting = choices.filter((c) => c.augmented && c.selectable && (statuses[`${dataset}/${c.id}`]?.state ?? 'idle') === 'idle')
  if (!waiting.length) return null
  const size = waiting.reduce((a, c) => a + c.bytes, 0)
  return (
    <button type="button" className="btn btn--quiet btn--small" onClick={() => onLoad(waiting.map((c) => c.id))}>
      {t('rec.pair.loadAll', { n: waiting.length, size: bytes(size) })}
    </button>
  )
}

export default function Recognise() {
  const { t } = useI18n()
  const { manifest, registry } = useBundle()
  const { dataset, model, setModel } = useChoice()
  const ds = manifest.datasets[dataset]
  const classes = ds.classes
  const choices = useMemo(() => modelChoices(ds), [ds])
  const digits = isDigitDataset(ds)
  const nKinds = useMemo(() => new Set(Object.values(manifest.datasets).flatMap((d) => Object.keys(d.models))).size, [manifest])
  const pairs = useMemo(() => modelPairs(choices), [choices])
  const paired = choices.some((c) => c.augmented)

  const padRef = useRef<DigitPadHandle>(null)
  const exploreRef = useRef<HTMLElement>(null)
  const [source, setSource] = useState<Source>({ kind: 'blank' })
  const pixels = pixelsOf(source)

  const statuses = useSyncExternalStore(registry.subscribe, registry.getSnapshot)
  const statusOf = (id: ModelId): ModelStatus => statuses[`${dataset}/${id}`] ?? { state: 'idle' }

  const touched = useRef(false)
  const loadEager = useCallback(() => {
    for (const c of choices) if (c.autoLoad && registry.status(dataset, c.id).state === 'idle') registry.load(dataset, c.id).catch(() => undefined)
  }, [choices, dataset, registry])
  const onIntent = useCallback(() => {
    touched.current = true
    loadEager()
  }, [loadEager])
  // a new dataset brings its own models: load the small ones once the visitor has started
  useEffect(() => {
    if (touched.current) loadEager()
  }, [loadEager])

  // a test image belongs to its dataset: switching datasets removes it
  useEffect(() => {
    if (source.kind === 'test' && source.dataset !== dataset) padRef.current?.clear()
  }, [dataset, source])

  const onPadChange = useCallback((c: PadChange) => {
    if (!c.image) {
      setSource({ kind: 'blank' })
      return
    }
    const pre = preprocessMnist(c.image)
    if (!pre) {
      setSource({ kind: 'empty' })
      return
    }
    setSource({ kind: 'pad', pre, crop: c.kind === 'draw' ? cropOf(pre.bbox, c.image.width, c.image.height) : null, upload: c.kind === 'upload' })
  }, [])

  const pickTest = useCallback(
    (image: TestImage) => {
      onIntent()
      padRef.current?.showImage(image.pixels)
      setSource({ kind: 'test', dataset, image })
    },
    [dataset, onIntent],
  )

  const execs = useAllExecutions(dataset, choices, pixels)
  const answers: Answer[] = choices.flatMap((c) => {
    const e = execs[c.id]
    return e.state === 'ready' ? [{ model: c.id, top: e.execution.top, score: e.execution.probs[e.execution.top] }] : []
  })
  const running = choices.filter((c) => c.selectable && pixels && (execs[c.id].state === 'running' || statusOf(c.id).state === 'loading')).length
  const verdict = consensusOf(answers)
  const verdictName = verdict ? classes[verdict.cls] : null
  const nameOf = (id: ModelId) => {
    const c = choices.find((x) => x.id === id)
    return c ? `${t(`model.${c.kind}`)}${c.augmented ? ` (${t('rec.pair.augmented')})` : ''}` : id
  }

  let say: string
  if (verdict && verdictName) {
    say =
      verdict.of === 1
        ? t('rec.consensus.single', { name: nameOf(answers[0].model), cls: verdictName })
        : verdict.votes === verdict.of
          ? t('rec.consensus.agree', { n: verdict.of, cls: verdictName })
          : t('rec.consensus.majority', { k: verdict.votes, n: verdict.of, cls: verdictName })
  } else if (source.kind === 'empty') say = t('rec.nothing.title')
  else if (pixels) say = t('rec.consensus.waiting')
  else say = digits ? t('rec.blank.title') : t('rec.blank.titleOther')

  let caption: string
  if (source.kind === 'test') caption = t('rec.test.caption', { title: ds.title })
  else if (source.kind === 'pad') caption = t('rec.prep.done', { x: source.pre.com.x.toFixed(1), y: source.pre.com.y.toFixed(1) })
  else if (source.kind === 'empty') caption = t('rec.nothing.text')
  else caption = digits ? t('rec.blank.text') : t('rec.blank.textOther')

  // screen-reader summary, throttled so live drawing does not chatter
  const [announce, setAnnounce] = useState('')
  const announceText = verdict && verdictName && running === 0 ? `${t('rec.consensus.title')}: ${verdictName}. ${say}` : ''
  useEffect(() => {
    const id = window.setTimeout(() => setAnnounce(announceText), 700)
    return () => window.clearTimeout(id)
  }, [announceText])

  const select = (id: ModelId) => {
    setModel(id)
    exploreRef.current?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' })
  }

  const testLabel = source.kind === 'test' ? classes[source.image.label] : null
  const renderCard = (c: ModelChoice, inPair: boolean) => {
    const e = execs[c.id]
    // in the paired grid the comparison that matters is within the row (marked there), not with the majority
    const dissent = !inPair && !!(e.state === 'ready' && verdict && verdict.of > 1 && e.execution.top !== verdict.cls)
    return (
      <ModelCard
        key={c.id}
        choice={c}
        dataset={dataset}
        classes={classes}
        exec={e}
        status={statusOf(c.id)}
        hasInput={!!pixels}
        dissent={dissent}
        selected={c.id === model}
        onSelect={() => select(c.id)}
        onLoad={() => {
          onIntent()
          registry.load(dataset, c.id).catch(() => undefined)
        }}
        paired={inPair}
      />
    )
  }
  return (
    <div className="console">
      <section className="intro">
        <h1 className="display">{t('rec.title', { n: nKinds })}</h1>
        <p className="lede">{t('rec.lede')}</p>
      </section>

      <section className="choose" aria-label={t('rec.ds.region')}>
        <DatasetPicker />
        <ReadingAs ds={ds} />
      </section>

      <div className="deck">
        <section className="screen" aria-labelledby="in-h">
          <header className="panel-head">
            <h2 id="in-h" className="panel-head__title">
              <span className="panel-head__no" aria-hidden="true">
                02
              </span>
              {t('rec.step.draw')}
            </h2>
          </header>
          <DigitPad ref={padRef} crop={source.kind === 'pad' ? source.crop : null} onChange={onPadChange} onIntent={onIntent} />
          {source.kind === 'test' && (
            <p className="screen__note num">{t('rec.test.shown', { index: source.image.index, name: testLabel ?? '' })}</p>
          )}
          {digits && (
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
          )}
          <TestImages dataset={dataset} ds={ds} chosen={source.kind === 'test' ? source.image : null} onPick={pickTest} onIntent={onIntent} />
        </section>

        <section className="readout" aria-labelledby="out-h">
          <header className="panel-head">
            <h2 id="out-h" className="panel-head__title">
              <span className="panel-head__no" aria-hidden="true">
                03
              </span>
              {t('rec.step.answer')}
            </h2>
            <span className="panel-head__tag">{t('rec.chip.dataset', { title: ds.title })}</span>
          </header>

          <div className={`verdict${verdict ? ' has-result' : ''}${running > 0 && verdict ? ' is-partial' : ''}`}>
            <div className="verdict__eye">
              <Mnist28 pixels={pixels} />
              <span className="verdict__eyelabel">{t('rec.step.see')}</span>
            </div>
            <div className="verdict__main">
              <p className="verdict__title">{t('rec.consensus.title')}</p>
              <p className={`verdict__digit${verdictName ? '' : ' is-none'}${verdictName && verdictName.length > 2 ? ' is-word' : ''}`} key={verdictName ?? 'none'} aria-hidden="true">
                {verdictName ?? '–'}
              </p>
              {testLabel && (
                <p className={`verdict__truth num${verdictName ? (verdictName === testLabel ? ' is-hit' : ' is-miss') : ''}`}>
                  {t('rec.test.truth', { name: testLabel })}
                  {verdictName ? ` · ${verdictName === testLabel ? t('rec.test.hit') : t('rec.test.miss')}` : ''}
                </p>
              )}
            </div>
            <div className="verdict__text">
              <p className="verdict__say">{say}</p>
              {running > 0 && verdict && <p className="caption verdict__pending">{t('rec.consensus.pending', { n: running })}</p>}
              <p className="caption">{caption}</p>
              {verdict?.tied && <p className="caption">{t('rec.consensus.tie')}</p>}
            </div>
          </div>

          {!digits && source.kind === 'pad' && (
            <p className="crossnote" role="note">
              <Warning size={16} weight="bold" aria-hidden="true" />
              <span>{t('rec.cross.drawn', { title: ds.title, n: classes.length, kind: kindLabel(t, ds) })}</span>
            </p>
          )}

          {paired ? (
            <div className="pairs">
              <div className="pairs__head">
                <span className="pairs__col" aria-hidden="true">
                  {t('rec.pair.augmented')}
                </span>
                <span className="pairs__col" aria-hidden="true">
                  {t('rec.pair.plain')}
                </span>
                <LoadVariants choices={choices} dataset={dataset} onLoad={(ids) => (onIntent(), ids.forEach((id) => registry.load(dataset, id).catch(() => undefined)))} />
              </div>
              <ul className="models models--paired">
                {pairs.map((p) => {
                  const a = p.augmented ? execs[p.augmented.id] : null
                  const b = p.plain ? execs[p.plain.id] : null
                  const differ = a?.state === 'ready' && b?.state === 'ready' && a.execution.top !== b.execution.top
                  return (
                    <li key={p.kind} className={`mpair${differ ? ' is-split' : ''}`} aria-label={t(`model.${p.kind}`)}>
                      <p className="mpair__name">
                        {t(`model.${p.kind}`)}
                        {differ && <span className="mpair__diff">{t('rec.pair.differ')}</span>}
                      </p>
                      <ul className="mpair__cells">
                        {[p.augmented, p.plain].map((c, i) => (c ? renderCard(c, true) : <li key={i} className="mcard mcard--none" aria-hidden="true" />))}
                      </ul>
                    </li>
                  )
                })}
              </ul>
            </div>
          ) : (
            <ul className="models">{choices.map((c) => renderCard(c, false))}</ul>
          )}
          <p className="visually-hidden" role="status" aria-live="polite">
            {announce}
          </p>
        </section>
      </div>

      <section className="explore-sec" ref={exploreRef} aria-labelledby="ex-h">
        <header className="explore-sec__head">
          <h2 id="ex-h" className="panel-head__title">
            <span className="panel-head__no" aria-hidden="true">
              04
            </span>
            {t('rec.explore.title')}
          </h2>
          <ModelPicker ds={ds} choices={choices} value={model} onChange={setModel} />
        </header>
        {model ? (
          <Suspense
            fallback={
              <div className="result__loading" aria-busy="true" aria-label={t('common.loadingPage')}>
                <div className="skeleton skeleton--block" />
              </div>
            }
          >
            <Explorer dataset={dataset} model={model} pixels={pixels} />
          </Suspense>
        ) : (
          <p className="caption">{t('rec.explore.none')}</p>
        )}
      </section>
    </div>
  )
}
