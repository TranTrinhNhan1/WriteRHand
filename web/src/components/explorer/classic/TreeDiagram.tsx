import { useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent } from 'react'
import { useI18n } from '../../../i18n'
import { subtreeSizes, type AdaBoostEnsemble, type Round } from '../../../engine/adaboost'
import { compactTree, fmt, rowCol, type CompactNode } from './model'

interface Props {
  ens: AdaBoostEnsemble
  /** tree index (0-based) */
  t: number
  /** route of the current input through this tree */
  round: Round
  classes: string[]
  width: number
  /** levels drawn completely; deeper levels only along the route */
  fullDepth: number
  /** stage the route depth by depth (off while playing fast and with reduced motion) */
  animate: boolean
}

const SLOT = 40
const ROW = 44
const NODE_W = 36
const NODE_H = 22

/**
 * One decision tree of the ensemble, compacted: the top levels in full, the input's route to its leaf,
 * and every other subtree folded into a stub that states how many real nodes it holds.
 */
export function TreeDiagram({ ens, t, round, classes, width, fullDepth, animate }: Props) {
  const { t: tr } = useI18n()
  const { mean, std, eps } = ens.preprocess
  const compact = useMemo(() => {
    const route = [...round.path.map((p) => p.node), round.leaf]
    return compactTree(ens.trees[t], subtreeSizes(ens, t), route, fullDepth, (v) => v * (std + eps) + mean)
  }, [ens, t, round, fullDepth, mean, std, eps])
  const [focus, setFocus] = useState<number | null>(null)
  const { nodes } = compact
  const W = Math.max(compact.slots * SLOT, NODE_W + 8)
  const H = (compact.maxDepth + 1) * ROW + 8
  const X = (n: CompactNode) => n.x * SLOT + SLOT / 2
  const Y = (n: CompactNode) => n.depth * ROW + 4
  const routeByNode = useMemo(() => new Map(round.path.map((p) => [p.node, p])), [round])
  const shown = focus != null && focus < nodes.length ? nodes[focus] : null
  const scroller = useRef<HTMLDivElement>(null)
  // keep the route in view when the drawing is wider than its box (small screens)
  const leafX = useMemo(() => {
    const leaf = nodes.find((n) => n.kind === 'leaf' && n.onRoute)
    return ((leaf ? leaf.x : 0) + (nodes[0]?.x ?? 0)) / 2 * SLOT + SLOT / 2
  }, [nodes])
  useEffect(() => {
    const el = scroller.current
    if (!el || el.scrollWidth <= el.clientWidth) return
    el.scrollLeft = Math.max(0, Math.min(el.scrollWidth - el.clientWidth, leafX - el.clientWidth / 2))
  }, [leafX, W])

  const describe = (n: CompactNode): string => {
    if (n.kind === 'stub') return tr('ex.ada.nodeStub', { n: n.count, d: n.depth })
    if (n.kind === 'leaf') return tr('ex.ada.nodeLeaf', { id: n.node, d: n.depth, name: classes[n.klass] })
    const { r, c } = rowCol(n.feature, width)
    const step = routeByNode.get(n.node)
    const base = tr('ex.ada.nodeSplit', { id: n.node, d: n.depth, r, c, th: fmt(n.thresholdPixel, 3), size: n.size })
    return step ? `${base} ${tr(step.goLeft ? 'ex.ada.nodeRouteL' : 'ex.ada.nodeRouteR', { v: fmt(step.pixel, 3) })}` : base
  }
  const onKey = (e: KeyboardEvent) => {
    const d: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }
    if (!d[e.key]) return
    e.preventDefault()
    setFocus((f) => (f == null ? 0 : (f + d[e.key] + nodes.length) % nodes.length))
  }
  const label = (n: CompactNode) => {
    if (n.kind === 'stub') return `+${n.count}`
    if (n.kind === 'leaf') return classes[n.klass].length > 6 ? `${classes[n.klass].slice(0, 5)}…` : classes[n.klass]
    const { r, c } = rowCol(n.feature, width)
    return `${r},${c}`
  }

  return (
    <div className="exc-tree">
      <div
        ref={scroller}
        className="exc-tree__scroll"
        tabIndex={0}
        role="group"
        aria-label={tr('ex.ada.treeAria', { t: t + 1, shown: compact.shown, total: compact.total })}
        onKeyDown={onKey}
        onBlur={() => setFocus(null)}
      >
        <svg key={t} viewBox={`0 0 ${W} ${H}`} width={W} height={H} className={animate ? 'exc-tree__svg is-anim' : 'exc-tree__svg'} aria-hidden="true">
          {nodes.map((n, i) => {
            if (n.parent < 0) return null
            const p = nodes[n.parent]
            const on = n.onRoute
            return (
              <g key={`e${i}`} className={on ? 'exc-edge is-route' : 'exc-edge'} style={{ '--d': n.depth } as CSSProperties}>
                <path d={`M${X(p)},${Y(p) + NODE_H}C${X(p)},${Y(p) + NODE_H + 12} ${X(n)},${Y(n) - 12} ${X(n)},${Y(n)}`} />
              </g>
            )
          })}
          {nodes.map((n, i) => (
            <g
              key={`n${i}`}
              className={`exc-node exc-node--${n.kind}${n.onRoute ? ' is-route' : ''}${focus === i ? ' is-focus' : ''}`}
              style={{ '--d': n.depth } as CSSProperties}
              transform={`translate(${X(n) - NODE_W / 2},${Y(n)})`}
              onPointerEnter={() => setFocus(i)}
              onClick={() => setFocus(i)}
            >
              <rect className="exc-node__hit" x={(NODE_W - SLOT) / 2} y={(NODE_H - 40) / 2} width={SLOT} height={40} />
              <rect className="exc-node__box" width={NODE_W} height={NODE_H} rx={n.kind === 'leaf' ? 11 : 3} />
              <text x={NODE_W / 2} y={NODE_H / 2 + 4} textAnchor="middle">
                {label(n)}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <p className="exc-tree__detail num" aria-live="polite">
        {shown ? describe(shown) : tr('ex.ada.treeHint')}
      </p>
      <p className="exc-fig__cap">{tr('ex.ada.treeCap', { shown: compact.shown, stubs: nodes.filter((n) => n.kind === 'stub').length, collapsed: compact.collapsed, total: compact.total, d: fullDepth + 1 })}</p>
    </div>
  )
}
