import { useEffect, useMemo, useState } from 'react'
import { useI18n } from '../../i18n'
import { usePlayback } from '../../engine/react'
import { useArchLink } from './ArchitectureDiagram'
import { HeatMap } from './HeatMap'
import { PlaybackBar } from './PlaybackBar'
import type { ExplorerViewProps } from './types'
import { DenseInspector } from './neural/DenseInspector'
import { fx, int, shape } from './neural/format'
import { NetworkGraph, type GraphColumn, type GraphEdge, type GraphSel } from './neural/NetworkGraph'
import { denseBlocks, gridDims, reachedLayers, selectShown, strongest, type Shown } from './neural/model'
import { useStepCaption } from './neural/steps'
import { Panel, PreprocessStrip, SoftmaxPanel, usePersisted } from './neural/ui'

export function useNarrow(q = '(max-width: 640px)'): boolean {
  const [v, setV] = useState(() => typeof window !== 'undefined' && window.matchMedia(q).matches)
  useEffect(() => {
    const mq = window.matchMedia(q)
    const on = () => setV(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [q])
  return v
}

const argmax = (v: ArrayLike<number>) => {
  let b = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i
  return b
}

/**
 * MLP: image -> flatten -> standardise -> dense/ReLU blocks (as many as the trace has) -> logits ->
 * softmax. Every value comes from the graph's runtime outputs; the selected neuron is rebuilt from
 * the parameters and compared with what the graph returned.
 */
export default function MlpView({ classes, loaded, execution }: ExplorerViewProps) {
  const { t } = useI18n()
  const { trace } = loaded
  const params = loaded.params!
  const out = execution.outputs
  const narrow = useNarrow()
  const blocks = useMemo(() => denseBlocks(trace), [trace])
  const std = trace.layers.find((l) => l.op === 'standardize')
  const image = trace.layers.find((l) => l.op === 'input')
  const flat = trace.layers.find((l) => l.op === 'flatten')
  const soft = trace.layers.find((l) => l.op === 'softmax')
  const side = image && image.out_shape.length === 2 ? image.out_shape[0] : Math.round(Math.sqrt(blocks[0].nIn))

  const pb = usePlayback(trace.layers.length, `${execution.key}/${execution.inputId}`, 520)
  const reachedIds = reachedLayers(trace, pb.step)
  const reached = useMemo(() => new Set(reachedIds), [reachedIds.join('|')]) // eslint-disable-line react-hooks/exhaustive-deps
  const arch = useArchLink(reachedIds)
  const caption = useStepCaption(trace, pb.step)

  // ------------------------------------------------------------------ columns of the graph
  // layer ids drawn by each column; the first one is the layer whose output the column shows
  const colLayers = useMemo(
    () => [[blocks[0].inputId, std?.id, flat?.id, image?.id].filter(Boolean) as string[], ...blocks.map((b) => [b.act?.id, b.dense.id].filter(Boolean) as string[])],
    [blocks, std, flat, image],
  )
  const values = useMemo(
    () => [out[blocks[0].inputRuntime ?? 'x'], ...blocks.map((b) => out[b.aRuntime ?? b.zRuntime ?? ''])],
    [out, blocks],
  )
  const [selRaw, setSel] = usePersisted<GraphSel | null>('mlp.sel', null)
  const [pins, setPins] = usePersisted<Record<number, number[]>>('mlp.pins', {})
  const sel: GraphSel = selRaw && selRaw.col < values.length && selRaw.neuron < values[selRaw.col].length ? selRaw : { col: Math.min(1, values.length - 1), neuron: argmax(values[Math.min(1, values.length - 1)]), from: null }

  const max = narrow ? 24 : 48
  const shown: Shown[] = useMemo(() => {
    const res: Shown[] = []
    const pinFor = (c: number) => [...(pins[c] ?? []), ...(sel.col === c ? [sel.neuron] : []), ...(sel.col === c + 1 && sel.from != null ? [sel.from] : [])]
    // hidden and output layers: most active first, plus evenly spaced ones
    for (let c = 1; c < values.length; c++) {
      const v = values[c]
      res[c] = selectShown(v.length, max, (i) => Math.abs(v[i]), Math.round(max / 6), pinFor(c))
    }
    // input: the pixels with the largest |x·w| into the shown first-layer neurons
    const b0 = blocks[0]
    const W0 = params.get(b0.weight)
    const x = values[0]
    const into = res[1].indices
    res[0] = selectShown(
      x.length,
      max,
      (i) => {
        let m = 0
        const row = i * b0.nOut
        for (const j of into) m = Math.max(m, Math.abs(x[i] * W0[row + j]))
        return m
      },
      Math.round(max / 6),
      pinFor(0),
    )
    return res
  }, [values, max, pins, sel.col, sel.neuron, sel.from, blocks, params])

  const K = narrow ? 4 : 6
  const edges: GraphEdge[][] = useMemo(() => {
    const res: GraphEdge[][] = [[]]
    for (let c = 1; c < values.length; c++) {
      const b = blocks[c - 1]
      const W = params.get(b.weight)
      const src = values[c - 1]
      const cand = shown[c - 1].indices
      const list: GraphEdge[] = []
      for (const j of shown[c].indices) {
        const v = (i: number) => src[i] * W[i * b.nOut + j]
        const take = sel.col === c && sel.neuron === j ? cand : strongest(cand, v, K)
        for (const i of take) list.push({ from: i, to: j, v: v(i) })
        if (sel.col === c && sel.neuron === j && sel.from != null && !take.includes(sel.from)) list.push({ from: sel.from, to: j, v: v(sel.from) })
      }
      res[c] = list
    }
    return res
  }, [values, blocks, params, shown, sel.col, sel.neuron, sel.from, K])

  const names = (c: number) => (c === 0 ? 'x' : blocks[c - 1].isOutput ? 'z' : `h${c}`)
  const columns: GraphColumn[] = values.map((v, c) => ({
    key: String(c),
    label: `${names(c)} · ${int(v.length)}`,
    values: v,
    shown: shown[c].indices,
    signed: c === 0 || !blocks[c - 1].act,
    names: c > 0 && blocks[c - 1].isOutput ? classes : undefined,
    filled: reached.has(colLayers[c][0]),
    wired: c > 0 && reached.has(blocks[c - 1].dense.id),
  }))

  // ------------------------------------------------------------------ diagram <-> selection
  const colOfLayer = (id: string | null) => {
    if (!id) return -1
    if (soft && id === soft.id) return values.length - 1
    return colLayers.findIndex((ids) => ids.includes(id))
  }
  useEffect(() => {
    const c = colOfLayer(arch.selected)
    if (c >= 0 && c !== sel.col) setSel({ col: c, neuron: argmax(values[c]), from: null })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arch.selected])
  useEffect(() => {
    if (colOfLayer(arch.selected) !== sel.col) arch.select(colLayers[sel.col][0])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel.col])

  const choose = (s: GraphSel) => setSel(s)
  const pin = (c: number, i: number) => {
    if (!shown[c].indices.includes(i)) setPins((p) => ({ ...p, [c]: [...(p[c] ?? []).filter((x) => x !== i), i].slice(-4) }))
  }

  // ------------------------------------------------------------------ inspector data
  const c = sel.col
  const block = c > 0 ? blocks[c - 1] : null
  const v = values[c]
  const active = useMemo(() => {
    let a = 0
    for (let i = 0; i < v.length; i++) if (v[i] > 0) a++
    return a
  }, [v])
  const grid = c === 0 ? { w: side, h: side } : gridDims(v.length)
  const lod = (s: Shown, c: number) =>
    s.total <= max
      ? t('ex.mlp.lodAll', { n: s.total })
      : t(c === 0 ? 'ex.mlp.lodInput' : 'ex.mlp.lod', { k: s.indices.length, n: int(s.total), top: s.byScore, even: s.strided, pin: s.pinned })
  const srcName = (b: (typeof blocks)[number]) => (i: number) =>
    b.index === 0 && b.nIn === side * side ? t('ex.mlp.pixel', { r: Math.floor(i / side), c: i % side }) : `${names(b.index)} #${i}`

  const layerTitle = (c: number) =>
    c === 0
      ? t('ex.mlp.colInput', { n: int(values[0].length) })
      : blocks[c - 1].isOutput
        ? t('ex.mlp.colOutput', { a: int(blocks[c - 1].nIn), b: blocks[c - 1].nOut })
        : t(blocks[c - 1].act ? 'ex.mlp.colHidden' : 'ex.mlp.colLinear', { k: c, a: int(blocks[c - 1].nIn), b: int(blocks[c - 1].nOut) })

  const preReached = (image && reached.has(image.id) ? 1 : 0) + (flat && reached.has(flat.id) ? 1 : 0) + (std && reached.has(std.id) ? 1 : 0)

  return (
    <div className="ex-view ex-mlp">
      <div className="ex-view__bar">
        <PlaybackBar pb={pb} caption={caption} unit={t('ex.shell.stepsUnit')} />
        <p className="visually-hidden" aria-live="polite">
          {pb.playing ? '' : caption}
        </p>
      </div>

      {std && (
        <Panel id="ex-mlp-pre" title={t('ex.mlp.preTitle')} sub={t('ex.mlp.preSub')}>
          <PreprocessStrip pixels={execution.pixels} x={values[0]} mean={std.mean!} std={std.std!} eps={std.eps ?? 0} flatten={!!flat} reached={preReached} side={side} />
        </Panel>
      )}

      <Panel id="ex-mlp-net" title={t('ex.mlp.netTitle')} sub={t('ex.mlp.netSub', { k: K })} className="ex-net">
        <NetworkGraph columns={columns} edges={edges} sel={sel} onSelect={choose} label={t('ex.mlp.graphAria', { n: columns.length })} />
        <ul className="ex-legend">
          <li>
            <span className="ex-swatch ex-swatch--pos" aria-hidden="true" />
            {t('ex.mlp.legendPos')}
          </li>
          <li>
            <span className="ex-swatch ex-swatch--neg" aria-hidden="true" />
            {t('ex.mlp.legendNeg')}
          </li>
          <li className="ex-legend__note">{t('ex.mlp.legendKeys')}</li>
        </ul>
        <ul className="ex-lod">
          {columns.map((col, i) => (
            <li key={col.key}>
              <b className="num">{names(i)}</b> {layerTitle(i)}: <span className="ex-lod__n">{lod(shown[i], i)}</span>
            </li>
          ))}
        </ul>
      </Panel>

      <div className="ex-inspect">
        <Panel id="ex-mlp-layer" title={layerTitle(c)} sub={c > 0 && blocks[c - 1].act ? t('ex.mlp.layerActive', { a: int(active), n: int(v.length) }) : t('ex.mlp.layerAll', { n: int(v.length) })}>
          <HeatMap
            values={v}
            width={grid.w}
            height={grid.h}
            mode={columns[c].signed ? 'diverging' : 'active'}
            label={t('ex.mlp.gridAria', { n: v.length, w: grid.w, h: grid.h })}
            selected={sel.neuron}
            onSelect={(i) => {
              if (i >= v.length) return
              pin(c, i)
              choose({ col: c, neuron: i, from: null })
            }}
            className="ex-layergrid"
          />
          <p className="ex-hint num">{t('ex.mlp.gridNote', { w: grid.w, h: grid.h, n: int(v.length) })}</p>
          <div className="ex-layerpick" role="group" aria-label={t('ex.mlp.pickLayer')}>
            {values.map((_, i) => (
              <button key={i} type="button" className="btn btn--quiet ex-chip" aria-pressed={i === c} onClick={() => choose({ col: i, neuron: argmax(values[i]), from: null })}>
                {names(i)} <span className="num">{int(values[i].length)}</span>
              </button>
            ))}
          </div>
        </Panel>

        <Panel
          id="ex-mlp-neuron"
          title={c === 0 ? t('ex.mlp.pixelTitle', { r: Math.floor(sel.neuron / side), c: sel.neuron % side }) : block!.isOutput ? t('ex.mlp.logitTitle', { k: classes[sel.neuron] ?? sel.neuron }) : t('ex.mlp.neuronTitle', { k: c, j: sel.neuron })}
          sub={c === 0 ? t('ex.mlp.pixelSub') : t('ex.mlp.neuronSub', { j: sel.neuron, a: int(block!.nIn) })}
          className="ex-neuron"
        >
          {c === 0 ? (
            <dl className="ex-terms">
              <div className="ex-term">
                <dt>p</dt>
                <dd className="num">{Math.round(execution.pixels[sel.neuron] * 255)}</dd>
              </div>
              <div className="ex-term ex-term--strong">
                <dt>x</dt>
                <dd className="num">{fx(values[0][sel.neuron], 4)}</dd>
              </div>
              <p className="ex-hint">{t('ex.mlp.pixelFeeds', { n: int(blocks[0].nOut) })}</p>
            </dl>
          ) : (
            <DenseInspector
              input={values[c - 1]}
              sym={c === 1 ? 'x' : 'a'}
              W={params.get(block!.weight)}
              b={params.get(block!.bias)}
              nIn={block!.nIn}
              nOut={block!.nOut}
              j={sel.neuron}
              zGraph={out[block!.zRuntime ?? ''][sel.neuron]}
              aGraph={block!.aRuntime ? out[block!.aRuntime][sel.neuron] : null}
              spatial={block!.index === 0 && block!.nIn === side * side ? { kind: 'image', side } : null}
              picked={sel.from}
              onPick={(i) => {
                if (i != null) pin(c - 1, i)
                choose({ ...sel, from: i })
              }}
              sourceName={srcName(block!)}
            />
          )}
        </Panel>
      </div>

      <SoftmaxPanel
        logits={out[blocks[blocks.length - 1].zRuntime ?? 'logits']}
        probs={execution.probs}
        classes={classes}
        revealed={!soft || reached.has(soft.id)}
        selected={sel.col === values.length - 1 ? sel.neuron : null}
        onSelect={(k) => choose({ col: values.length - 1, neuron: k, from: null })}
      />
      <p className="ex-hint num">{t('ex.mlp.shapes', { s: trace.layers.map((l) => shape(l.out_shape)).join(' → ') })}</p>
    </div>
  )
}
