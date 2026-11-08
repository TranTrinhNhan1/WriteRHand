import { ArrowClockwise, ClockCounterClockwise, Lightning, PencilSimpleLine } from '@phosphor-icons/react'
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { useI18n } from '../../i18n'
import { isAugmented, kindOf, type ModelId, type ModelKind } from '../../engine/contract'
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
  // same layout as softmax regression (one dense layer of per-class weights); its output is scores
  svm: SoftmaxView,
  adaboost: AdaBoostView,
}

interface Props {
  dataset: string
  model: ModelId
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

  const tabs: { id: Mode; icon: ReactNode; title: string }[] = [
    { id: 'inference', icon: <Lightning size={14} weight="fill" aria-hidden="true" />, title: t('ex.shell.mode.inference') },
    { id: 'replay', icon: <ClockCounterClockwise size={14} aria-hidden="true" />, title: t('ex.shell.mode.replay') },
  ]
  // an augmented variant has no recorded training run: no replay tab for it
  const shownTabs = isAugmented(entry) && !entry.replay ? tabs.filter((x) => x.id !== 'replay') : tabs
  const onTabKey = (e: KeyboardEvent) => {
    if ((e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') || shownTabs.length < 2) return
    e.preventDefault()
    const next: Mode = mode === 'inference' ? 'replay' : 'inference'
    setMode(next)
    document.getElementById(`${uid}-tab-${next}`)?.focus()
  }

  return (
    <section className="ex" aria-label={`${t(`model.${kindOf(entry)}`)}${isAugmented(entry) ? ` · ${t('rec.pair.augmented')}` : ''}`} data-mode={mode}>
      <div className="ex-top">
        <div className="ex-modes" role="tablist" aria-label={t('ex.shell.mode.label')}>
          {shownTabs.map((tab) => (
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
              {tab.icon}
              {tab.title}
            </button>
          ))}
        </div>
        <SpecLine dataset={dataset} model={model} />
      </div>

      <div className="ex-panelwrap" id={`${uid}-panel`} role="tabpanel" aria-labelledby={`${uid}-tab-${mode}`} data-mode={mode}>
        {mode === 'inference' ? (
          <PersistContext.Provider value={persist}>
            <ArchLinkContext.Provider value={link}>
              <LiveLine exec={exec} pending={state.state === 'running' || state.state === 'loading'} />
              <ArchitectureDiagram trace={trace} reached={exec ? reachedSet : null} selected={archSel} onSelect={(id) => setArchSel(id)} />
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

// --------------------------------------------------------------------------- what you are looking at
function SpecLine({ dataset, model }: { dataset: string; model: ModelId }) {
  const { t } = useI18n()
  const { manifest } = useBundle()
  const entry = manifest.datasets[dataset].models[model]!
  const v = entry.validation
  const trace = entry.trace
  return (
    <div className="ex-spec">
      <dl className="ex-spec__grid">
        <div className="ex-spec__row">
          <dt>{t('ex.shell.spec.params')}</dt>
          <dd className="num">{trace.n_parameters != null ? int(trace.n_parameters) : t('ex.shell.spec.paramsNone')}</dd>
        </div>
        <div className="ex-spec__row">
          <dt>{t('ex.shell.spec.files')}</dt>
          <dd className="num">{`${entry.format === 'onnx' ? 'ONNX' : t('ex.shell.spec.binary')} ${bytes(entry.size_bytes)}${trace.params ? ` + ${t('ex.shell.spec.paramFile')} ${bytes(trace.params.nbytes)}` : ''}`}</dd>
        </div>
        <div className="ex-spec__row" data-ok={v.ok}>
          <dt>{t('ex.shell.spec.validation')}</dt>
          <dd className="num">{t(v.ok ? 'ex.shell.spec.valOk' : 'ex.shell.spec.valBad', { d: diff(v.max_abs_prob_diff), n: int(v.n_checked), agree: pct(v.argmax_agreement, 1) })}</dd>
        </div>
      </dl>
      <details className="ex-spec__more">
        <summary>{t('ex.shell.spec.sha')}</summary>
        <dl className="ex-spec__grid">
          <div className="ex-spec__row">
            <dt>{t('ex.shell.spec.sha')}</dt>
            <dd className="num">
              <code>{entry.sha256}</code>
            </dd>
          </div>
          <div className="ex-spec__row">
            <dt>{t('ex.shell.spec.bundle')}</dt>
            <dd className="num">
              <code>{manifest.bundle_id}</code>
            </dd>
          </div>
          <div className="ex-spec__row">
            <dt>{t('ex.shell.spec.schema')}</dt>
            <dd className="num">
              <code>{trace.schema}</code>
            </dd>
          </div>
        </dl>
      </details>
    </div>
  )
}

function LiveLine({ exec, pending }: { exec: Execution | null; pending: boolean }) {
  const { t } = useI18n()
  return (
    <p className="ex-live num" data-state={exec ? 'ready' : pending ? 'pending' : 'idle'}>
      {exec ? t('ex.shell.badgeLiveMeta', { id: exec.inputId, ms: exec.ms < 0.05 ? '<0.1' : exec.ms.toFixed(1) }) : pending ? t('ex.shell.badgePending') : t('ex.shell.badgeIdle')}
    </p>
  )
}

// --------------------------------------------------------------------------- inference states
function Inference({ dataset, model, state, onRetry }: { dataset: string; model: ModelId; state: ReturnType<typeof useExecution>; onRetry: () => void }) {
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
