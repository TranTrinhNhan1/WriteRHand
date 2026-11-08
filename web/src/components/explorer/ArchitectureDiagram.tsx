import { createContext, useContext, useEffect, type CSSProperties } from 'react'
import { useI18n, type Key } from '../../i18n'
import type { Trace, TraceLayer } from '../../engine/contract'
import { fx, int, shape } from './neural/format'
import { layerParamCount } from './neural/model'

interface Props {
  trace: Trace
  /** Layer ids lit up so far (playback) and the layer the user selected. Without `reached` every layer is lit. */
  reached?: Set<string> | null
  selected?: string | null
  onSelect?: (layerId: string) => void
}

// --------------------------------------------------------------------------- link between the shell and a view
/**
 * The Explorer shell draws the diagram; a view that plays the computation tells it which layers are
 * done (`reached`) and receives clicks on a node (`selected`). Views that do not use it simply get a
 * fully lit diagram.
 */
export interface ArchLink {
  selected: string | null
  select: (id: string | null) => void
  setReached: (ids: string[] | null) => void
}

export const ArchLinkContext = createContext<ArchLink | null>(null)

/** Call from a view: `reached` = ids computed so far (null = all). Returns the layer chosen on the diagram. */
export function useArchLink(reached: string[] | null): { selected: string | null; select: (id: string | null) => void } {
  const link = useContext(ArchLinkContext)
  const key = reached ? reached.join('|') : null
  useEffect(() => {
    link?.setReached(key == null ? null : key === '' ? [] : key.split('|'))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, link?.setReached])
  useEffect(() => () => link?.setReached(null), [link?.setReached]) // eslint-disable-line react-hooks/exhaustive-deps
  return { selected: link?.selected ?? null, select: link?.select ?? (() => undefined) }
}

// --------------------------------------------------------------------------- one node
function detail(l: TraceLayer, t: (k: Key, p?: Record<string, string | number>) => string): string {
  switch (l.op) {
    case 'dense':
      return `${int(l.in_features!)} → ${int(l.out_features!)}`
    case 'conv2d': {
      const [cout, cin, kh, kw] = l.kernel_shape!
      return t('ex.arch.convDetail', { k: `${kh}×${kw}`, cin, cout, s: l.stride ?? 1 })
    }
    case 'maxpool2d':
      return t('ex.arch.poolDetail', { k: `${l.window}×${l.window}`, s: l.stride ?? l.window! })
    case 'standardize':
      return t('ex.arch.stdDetail', { mean: fx(l.mean!, 4), std: fx(l.std!, 4) })
    case 'tree_ensemble': {
      const d = l.depths ?? []
      const maxD = d.length ? Math.max(...d) : 0
      return t('ex.arch.treesDetail', { n: int(l.n_estimators ?? 0), d: maxD })
    }
    case 'gaussian_loglik':
      return t('ex.arch.nbDetail', { k: l.n_classes ?? 0, f: int(l.n_features ?? 0) })
    case 'relu':
      return 'max(0, z)'
    case 'softmax':
      return 'e^z / Σ e^z'
    case 'weighted_vote':
      return t('ex.arch.voteDetail')
    case 'normalize':
      return t('ex.arch.normDetail')
    default:
      return ''
  }
}

/** A size glyph: vectors are bars whose length grows with log(size), maps are (stacked) squares. The real size is always written next to it. */
function Glyph({ s }: { s: number[] }) {
  if (s.length === 1) {
    const w = Math.max(6, Math.min(100, (Math.log2(Math.max(2, s[0])) / Math.log2(1024)) * 100))
    return <span className="ex-arch__glyph ex-arch__glyph--vec" style={{ '--w': `${w}%` } as CSSProperties} aria-hidden="true" />
  }
  const side = s[s.length - 1]
  const stack = s.length === 3 ? Math.min(3, s[0]) : 1
  const px = Math.max(10, Math.min(30, Math.round(10 + (side / 28) * 20)))
  return (
    <span className="ex-arch__glyph ex-arch__glyph--map" style={{ '--px': `${px}px`, '--stack': stack } as CSSProperties} aria-hidden="true">
      {Array.from({ length: stack }, (_, i) => (
        <i key={i} style={{ '--i': i } as CSSProperties} />
      ))}
    </span>
  )
}

/**
 * Architecture drawn from `trace.layers` ONLY, in manifest (computation) order: operator, output
 * shape, operator settings and the number of learned values where the parameter index has them.
 * Works for any number of layers and any op of the trace contract.
 */
export function ArchitectureDiagram({ trace, reached, selected, onSelect }: Props) {
  const { t } = useI18n()
  const ids = trace.layers.map((l) => l.id)
  const lastReached = reached ? Math.max(-1, ...ids.map((id, i) => (reached.has(id) ? i : -1))) : -1
  return (
    <ol className="ex-arch" aria-label={t('ex.arch.label', { n: trace.layers.length })}>
      {trace.layers.map((l, i) => {
        const params = layerParamCount(trace, l)
        const nodes = l.op === 'tree_ensemble' ? (l.node_counts ?? []).reduce((a, b) => a + b, 0) : null
        const isReached = !reached || reached.has(l.id)
        const prevId = i > 0 ? ids[i - 1] : null
        const branch = l.inputs.length > 0 && l.inputs[0] !== prevId ? l.inputs[0] : null
        const opName = t(`ex.arch.op.${l.op}` as Key)
        const d = detail(l, t)
        const extra = params != null ? t('ex.arch.params', { n: int(params) }) : nodes != null ? t('ex.arch.nodes', { n: int(nodes) }) : null
        return (
          <li key={l.id} className="ex-arch__item" style={{ '--i': i } as CSSProperties}>
            {i > 0 && (
              <span className="ex-arch__wire" aria-hidden="true">
                {branch ? `↳ ${branch}` : ''}
              </span>
            )}
            <button
              type="button"
              className="ex-arch__node"
              data-reached={isReached ? 'true' : 'false'}
              data-now={reached && i === lastReached ? 'true' : undefined}
              data-op={l.op}
              aria-pressed={selected === l.id}
              onClick={() => onSelect?.(l.id)}
              title={[d, extra].filter(Boolean).join(' · ') || undefined}
              aria-label={`${i + 1}. ${opName} ${l.id}, ${t('ex.arch.out')} ${shape(l.out_shape)}${d ? `, ${d}` : ''}${extra ? `, ${extra}` : ''}${isReached ? '' : `, ${t('ex.arch.pending')}`}`}
            >
              <span className="ex-arch__head">
                <span className="num ex-arch__idx">{String(i + 1).padStart(2, '0')}</span>
                <span className="ex-arch__op">{opName}</span>
              </span>
              <Glyph s={l.out_shape} />
              <span className="num ex-arch__shape">{shape(l.out_shape)}</span>
              {d && <span className="ex-arch__detail">{d}</span>}
              {extra && <span className="num ex-arch__params">{extra}</span>}
            </button>
          </li>
        )
      })}
    </ol>
  )
}
