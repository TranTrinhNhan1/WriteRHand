import { ArrowClockwise, CaretDown, ClockCounterClockwise, Lightning, PencilSimpleLine } from '@phosphor-icons/react'
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { useI18n } from '../../i18n'
import type { ModelKind } from '../../engine/contract'
import { useBundle, useExecution } from '../../engine/react'
import type { Execution } from '../../engine/store'
import { bytes, pct } from '../../lib/format'
import AdaBoostView from './AdaBoostView'
import { ArchitectureDiagram, ArchLinkContext, type ArchLink } from './ArchitectureDiagram'
import CnnView from './CnnView'
import MlpView from './MlpView'
import NaiveBayesView from './NaiveBayesView'
import ReplayPanel from './ReplayPanel'
import SoftmaxView from './SoftmaxView'
import type { ExplorerViewProps } from './types'
import { diff, int } from './neural/format'
import { PersistContext } from './neural/ui'

type Mode = 'inference' | 'replay'

const VIEWS: Record<ModelKind, (p: ExplorerViewProps) => ReactNode> = {
  mlp: MlpView,
  cnn: CnnView,
  naive_bayes: NaiveBayesView,
  softmax: SoftmaxView,
  adaboost: AdaBoostView,
}

interface Props {
  dataset: string
  model: ModelKind
  pixels: Float32Array | null
}

/**
 * Model Execution Explorer: shows the real computation of the deployed model of `dataset` on
 * `pixels` (inference), or the recorded run that trained it (training replay). It never shows a
 * result that does not belong to exactly this model and this input.
 */
export default function Explorer({ dataset, model, pixels }: Props) {
  const [attempt, setAttempt] = useState(0)
  return <ExplorerBody key={`${dataset}/${model}/${attempt}`} dataset={dataset} model={model} pixels={pixels} onRetry={() => setAttempt((a) => a + 1)} />
}

function ExplorerBody({ dataset, model, pixels, onRetry }: Props & { onRetry: () => void }) {
  const { t } = useI18n()
  const { manifest, registry } = useBundle()
  const ds = manifest.datasets[dataset]
  const entry = ds?.models[model]
  const [mode, setMode] = useState<Mode>('inference')
  const state = useExecution(dataset, model, pixels, mode === 'inference')
  const persist = useMemo(() => new Map<string, unknown>(), [])
  const [reached, setReachedState] = useState<string[] | null>(null)
  const [archSel, setArchSel] = useState<string | null>(null)
  const setReached = useCallback((ids: string[] | null) => setReachedState(ids), [])
  const link = useMemo<ArchLink>(() => ({ selected: archSel, select: setArchSel, setReached }), [archSel, setReached])
  const reachedSet = useMemo(() => (reached ? new Set(reached) : null), [reached])
  const uid = useId()

  if (!ds || !entry) {
    return (
      <section className="ex">
        <p className="msg">{t('ex.shell.missing')}</p>
      </section>
    )
  }
  const trace = entry.trace
  const exec = state.state === 'ready' ? state.execution : null
  const retry = () => {
    registry.load(dataset, model).catch(() => undefined)
    onRetry()
  }

  const tabs: { id: Mode; icon: ReactNode; title: string; sub: string }[] = [
    { id: 'inference', icon: <Lightning size={18} weight="fill" aria-hidden="true" />, title: t('ex.shell.mode.inference'), sub: t('ex.shell.mode.inferenceSub') },
    { id: 'replay', icon: <ClockCounterClockwise size={18} aria-hidden="true" />, title: t('ex.shell.mode.replay'), sub: t('ex.shell.mode.replaySub') },
  ]
  const onTabKey = (e: KeyboardEvent) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    const next: Mode = mode === 'inference' ? 'replay' : 'inference'
    setMode(next)
    document.getElementById(`${uid}-tab-${next}`)?.focus()
  }

  return (
    <section className="ex" aria-labelledby={`${uid}-title`} data-mode={mode}>
      <header className="ex__head">
        <p className="eyebrow">{t('ex.shell.eyebrow')}</p>
        <h2 className="h2" id={`${uid}-title`}>
          {t('ex.shell.title')}
        </h2>
        <p className="lede ex__lede">{t('ex.shell.lede')}</p>
      </header>

      <SpecSheet dataset={dataset} model={model} exec={exec} />

      <div className="ex-modes" role="tablist" aria-label={t('ex.shell.mode.label')}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            id={`${uid}-tab-${tab.id}`}
            type="button"
            role="tab"
            className="ex-mode"
            data-mode={tab.id}
            aria-selected={mode === tab.id}
            aria-controls={`${uid}-panel`}
            tabIndex={mode === tab.id ? 0 : -1}
            onClick={() => setMode(tab.id)}
            onKeyDown={onTabKey}
          >
            <span className="ex-mode__icon">{tab.icon}</span>
            <span className="ex-mode__text">
              <span className="ex-mode__title">{tab.title}</span>
              <span className="ex-mode__sub">{tab.sub}</span>
            </span>
          </button>
        ))}
      </div>

      <div className="ex-panelwrap" id={`${uid}-panel`} role="tabpanel" aria-labelledby={`${uid}-tab-${mode}`} data-mode={mode}>
        {mode === 'inference' ? (
          <PersistContext.Provider value={persist}>
            <ArchLinkContext.Provider value={link}>
              <LiveBadge exec={exec} pending={state.state === 'running' || state.state === 'loading'} />
              <HowToRead />
              <div className="ex-archwrap">
                <h3 className="ex-sub">{t('ex.arch.title')}</h3>
                <ArchitectureDiagram trace={trace} reached={exec ? reachedSet : null} selected={archSel} onSelect={(id) => setArchSel(id)} />
                <p className="ex-hint">{t('ex.arch.note')}</p>
              </div>
              <Inference dataset={dataset} model={model} state={state} onRetry={retry} />
            </ArchLinkContext.Provider>
          </PersistContext.Provider>
        ) : (
          <ReplayPanel dataset={dataset} model={model} />
        )}
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------- header: what you are looking at
function SpecSheet({ dataset, model, exec }: { dataset: string; model: ModelKind; exec: Execution | null }) {
  const { t } = useI18n()
  const { manifest } = useBundle()
  const ds = manifest.datasets[dataset]
  const entry = ds.models[model]!
  const v = entry.validation
  const trace = entry.trace
  const rows: { k: string; v: ReactNode; title?: string }[] = [
    { k: t('ex.shell.spec.model'), v: entry.title },
    { k: t('ex.shell.spec.dataset'), v: ds.title },
    { k: t('ex.shell.spec.params'), v: trace.n_parameters != null ? int(trace.n_parameters) : t('ex.shell.spec.paramsNone') },
    {
      k: t('ex.shell.spec.files'),
      v: `${entry.format === 'onnx' ? 'ONNX' : t('ex.shell.spec.binary')} ${bytes(entry.size_bytes)}${trace.params ? ` + ${t('ex.shell.spec.paramFile')} ${bytes(trace.params.nbytes)}` : ''}`,
    },
    {
      k: t('ex.shell.spec.validation'),
      v: t(v.ok ? 'ex.shell.spec.valOk' : 'ex.shell.spec.valBad', { d: diff(v.max_abs_prob_diff), n: int(v.n_checked), agree: pct(v.argmax_agreement, 1) }),
    },
    { k: t('ex.shell.spec.sha'), v: <code title={entry.sha256}>{entry.sha256.slice(0, 12)}…</code> },
    { k: t('ex.shell.spec.bundle'), v: <code title={manifest.bundle_id}>{manifest.bundle_id.slice(0, 8)}</code> },
    { k: t('ex.shell.spec.schema'), v: <code>{trace.schema}</code> },
    { k: t('ex.shell.spec.input'), v: exec ? <code>{exec.inputId}</code> : <span className="ex-muted">{t('ex.shell.spec.noInput')}</span> },
  ]
  return (
    <div className="ex-spec">
      <dl className="ex-spec__grid">
        {rows.map((r) => (
          <div key={r.k} className="ex-spec__row">
            <dt>{r.k}</dt>
            <dd className="num">{r.v}</dd>
          </div>
        ))}
      </dl>
      <div className="ex-spec__classes">
        <span className="ex-spec__k">{t('ex.shell.spec.classes', { n: ds.classes.length })}</span>
        <ol className="ex-classes">
          {ds.classes.map((c, i) => (
            <li key={i}>
              {c !== String(i) && <span className="num">{i}</span>} {c}
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

function LiveBadge({ exec, pending }: { exec: Execution | null; pending: boolean }) {
  const { t } = useI18n()
  return (
    <p className="ex-badge ex-badge--live" data-state={exec ? 'ready' : pending ? 'pending' : 'idle'}>
      <Lightning size={14} weight="fill" aria-hidden="true" />
      <b>{t('ex.shell.badgeLive')}</b>
      {exec && <span className="num">{t('ex.shell.badgeLiveMeta', { id: exec.inputId, ms: exec.ms < 0.05 ? '<0.1' : exec.ms.toFixed(1) })}</span>}
      {!exec && <span>{pending ? t('ex.shell.badgePending') : t('ex.shell.badgeIdle')}</span>}
    </p>
  )
}

function HowToRead() {
  const { t, tl } = useI18n()
  return (
    <details className="ex-howto">
      <summary>
        <CaretDown size={14} aria-hidden="true" className="ex-howto__caret" />
        {t('ex.shell.howto.title')}
      </summary>
      <ul>
        {tl('ex.shell.howto.items').map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ul>
    </details>
  )
}

// --------------------------------------------------------------------------- inference states
function Inference({ dataset, model, state, onRetry }: { dataset: string; model: ModelKind; state: ReturnType<typeof useExecution>; onRetry: () => void }) {
  const { t } = useI18n()
  const { manifest, registry } = useBundle()
  const ds = manifest.datasets[dataset]
  const entry = ds.models[model]!
  const box = useRef<HTMLDivElement>(null)
  const lastH = useRef(0)
  useLayoutEffect(() => {
    if (state.state === 'ready' && box.current) lastH.current = box.current.getBoundingClientRect().height
  })
  useEffect(() => {
    const el = box.current
    if (!el || state.state !== 'ready') return
    const ro = new ResizeObserver(() => (lastH.current = el.getBoundingClientRect().height))
    ro.observe(el)
    return () => ro.disconnect()
  }, [state.state])

  if (state.state === 'idle') {
    return (
      <div className="ex-state ex-state--empty">
        <PencilSimpleLine size={28} aria-hidden="true" />
        <p className="label">{t('ex.shell.empty.title')}</p>
        <p className="caption">{t('ex.shell.empty.text')}</p>
      </div>
    )
  }
  if (state.state === 'error') {
    return (
      <div className="ex-state ex-state--error" role="alert">
        <p className="msg msg--error">{t(state.error.key, state.error.params)}</p>
        <button type="button" className="btn btn--quiet" onClick={onRetry}>
          <ArrowClockwise size={16} aria-hidden="true" />
          {t('common.retry')}
        </button>
      </div>
    )
  }
  if (state.state !== 'ready') {
    const size = entry.size_bytes + (entry.trace.params?.nbytes ?? 0)
    return (
      <div className="ex-state ex-state--pending" aria-busy="true" style={{ minHeight: lastH.current || undefined }}>
        <p className="caption" role="status">
          {state.state === 'loading' ? t('ex.shell.loading', { size: bytes(size) }) : t('ex.shell.running')}
        </p>
        <div className="skeleton skeleton--line" />
        <div className="skeleton skeleton--line skeleton--short" />
        <div className="skeleton skeleton--block" />
      </div>
    )
  }
  const loaded = registry.get(dataset, model)
  if (!loaded) return null
  const View = VIEWS[loaded.trace.kind]
  return (
    <div ref={box} className="ex-viewwrap">
      <View dataset={dataset} classes={ds.classes} loaded={loaded} execution={state.execution} />
    </div>
  )
}
