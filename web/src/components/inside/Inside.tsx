import { useEffect, useId, useRef, useState } from 'react'
import { useI18n } from '../../i18n'
import { useModelStatuses, useModelStore, useSiteData } from '../../lib/context'
import { bytes } from '../../lib/format'
import type { Trace } from '../../lib/trace'
import type { ModelId } from '../../lib/types'
import { AdaView } from './AdaView'
import { ClassMaps } from './ClassMaps'
import { MlpView } from './MlpView'

function softmax(z: ArrayLike<number>): Float64Array {
  let m = -Infinity
  for (let i = 0; i < z.length; i++) m = Math.max(m, z[i])
  const e = Float64Array.from(z as ArrayLike<number> as number[], (v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** "Look inside": what each algorithm does with the drawing, step by step. */
export function Inside({ pixels }: { pixels: Float32Array | null }) {
  const { t } = useI18n()
  const { manifest } = useSiteData()
  const store = useModelStore()
  const statuses = useModelStatuses(store)
  const ids = store.ids()
  const [sel, setSel] = useState<ModelId>(ids.includes('mlp') ? 'mlp' : ids[0])
  const [trace, setTrace] = useState<{ id: ModelId; data: Trace | null } | null>(null)
  const [play, setPlay] = useState(1)
  const tabs = useRef<Record<string, HTMLButtonElement | null>>({})
  const uid = useId()

  const status = statuses[sel]
  const ready = status?.state === 'ready'

  // The unrolling animation replays whenever a new drawing starts.
  const wasBlank = useRef(true)
  useEffect(() => {
    if (pixels && wasBlank.current) setPlay((p) => p + 1)
    wasBlank.current = !pixels
  }, [pixels])

  useEffect(() => {
    if (!pixels || !ready) {
      setTrace(null)
      return
    }
    let cancelled = false
    store.trace(sel, pixels).then(
      (data) => !cancelled && setTrace({ id: sel, data }),
      () => !cancelled && setTrace({ id: sel, data: null }),
    )
    return () => {
      cancelled = true
    }
  }, [pixels, ready, sel, store])

  const onKey = (e: React.KeyboardEvent) => {
    const i = ids.indexOf(sel)
    const next = e.key === 'ArrowRight' ? ids[(i + 1) % ids.length] : e.key === 'ArrowLeft' ? ids[(i + ids.length - 1) % ids.length] : null
    if (!next) return
    e.preventDefault()
    setSel(next)
    if (statuses[next]?.state !== 'ready') void store.load(next)
    tabs.current[next]?.focus()
  }

  const name = t(`model.${sel}`)
  const size = bytes(manifest.models[sel]?.size_bytes)
  const data = trace && trace.id === sel ? trace.data : null

  let body: React.ReactNode
  if (!ready) {
    body =
      status?.state === 'loading' ? (
        <p className="inside__msg">{t('inside.loading', { name })}</p>
      ) : (
        <div className="inside__msg">
          <p>{t('inside.notLoaded', { name })}</p>
          <button type="button" className="btn btn--primary" onClick={() => void store.load(sel)}>
            {status?.state === 'error' ? t('common.retry') : t('rec.card.load', { size })}
          </button>
        </div>
      )
  } else if (!pixels) {
    body = <p className="inside__msg">{t('inside.empty', { name })}</p>
  } else if (!data) {
    body = <p className="inside__msg">{trace && trace.id === sel ? t('inside.unavailable') : t('inside.loading', { name })}</p>
  } else if (data.kind === 'mlp') {
    body = <MlpView trace={data} probs={softmax(data.logits)} pixels={pixels} play={play} onReplay={() => setPlay((p) => p + 1)} />
  } else if (data.kind === 'adaboost') {
    body = <AdaView trace={data} pixels={pixels} />
  } else {
    body = <ClassMaps trace={data} probs={softmax(data.logits)} pixels={pixels} />
  }

  return (
    <section className="inside" aria-labelledby="inside-h">
      <header className="panel-head">
        <h2 id="inside-h" className="panel-head__title">
          <span className="panel-head__no" aria-hidden="true">
            03
          </span>
          {t('inside.title')}
        </h2>
      </header>
      <p className="lede inside__lede">{t('inside.lede')}</p>
      <div className="inside__tabs" role="tablist" aria-label={t('inside.tabs')} onKeyDown={onKey}>
        {ids.map((id) => (
          <button
            key={id}
            ref={(el) => {
              tabs.current[id] = el
            }}
            type="button"
            role="tab"
            id={`${uid}-tab-${id}`}
            aria-selected={id === sel}
            aria-controls={`${uid}-panel`}
            tabIndex={id === sel ? 0 : -1}
            className="inside__tab"
            onClick={() => {
              setSel(id)
              if (statuses[id]?.state !== 'ready') void store.load(id)
            }}
          >
            {t(`model.${id}`)}
          </button>
        ))}
      </div>
      <div className="inside__panel" role="tabpanel" id={`${uid}-panel`} aria-labelledby={`${uid}-tab-${sel}`}>
        <p className="inside__intro">{t(`inside.${sel}.intro`)}</p>
        {body}
      </div>
    </section>
  )
}
