import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react'
import { useI18n, type Key } from '../../i18n'
import { usePlayback, type Playback } from '../../engine/react'
import { convPatch, poolWindow, receptiveField } from '../../engine/reconstruct'
import { useArchLink } from './ArchitectureDiagram'
import { HeatMap } from './HeatMap'
import { PlaybackBar } from './PlaybackBar'
import type { ExplorerViewProps } from './types'
import { DenseInspector, type SpatialInput } from './neural/DenseInspector'
import { fx, int, shape } from './neural/format'
import { channel, channelArgmax, convStages, denseBlocks, kernelSlice, poolCell, pooledReady, rasterPos, reachedLayers, runtimeOf, type ConvStage } from './neural/model'
import { useStepCaption } from './neural/steps'
import { CheckRow, Hint, Panel, PreprocessStrip, SoftmaxPanel, Term, usePersisted } from './neural/ui'
import Net3D from './three/Net3D'
import { NetSwitch } from './three/NetSwitch'
import { cnnSpec } from './three/spec'
import type { Pick3D } from './three/engine3d'

const argmax = (v: ArrayLike<number>) => {
  let b = 0
  for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i
  return b
}

const absMax = (v: ArrayLike<number>) => {
  let m = 0
  for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]))
  return m || 1
}

/** Copy of `v` with the cells at or after `computed` blanked (not computed yet during the sliding playback). */
function reveal(v: Float32Array, computed: number | null): Float32Array {
  if (computed == null || computed >= v.length) return v
  const out = new Float32Array(v.length)
  out.set(v.subarray(0, Math.max(0, computed)))
  return out
}

/**
 * CNN: input -> (conv -> ReLU -> max pool) stages -> flatten -> dense -> softmax, all read from the
 * trace. Kernels come from the parameter file, feature maps from the graph's runtime outputs; the
 * selected output location is rebuilt (patch × kernel, sum, bias, ReLU, pooling) and compared.
 */
export default function CnnView({ classes, loaded, execution }: ExplorerViewProps) {
  const { t } = useI18n()
  const { trace } = loaded
  const out = execution.outputs
  const stages = useMemo(() => convStages(trace), [trace])
  const blocks = useMemo(() => denseBlocks(trace), [trace])
  const std = trace.layers.find((l) => l.op === 'standardize')
  const image = trace.layers.find((l) => l.op === 'input')
  const soft = trace.layers.find((l) => l.op === 'softmax')
  const side = image && image.out_shape.length === 2 ? image.out_shape[0] : 28

  // ------------------------------------------------------------------ forward pass, layer by layer
  const pb = usePlayback(trace.layers.length, `${execution.key}/${execution.inputId}`, 480)
  const reachedIds = reachedLayers(trace, pb.step)
  const reached = useMemo(() => new Set(reachedIds), [reachedIds.join('|')]) // eslint-disable-line react-hooks/exhaustive-deps
  const arch = useArchLink(reachedIds)
  const caption = useStepCaption(trace, pb.step)

  const [stageIdx, setStageIdx] = usePersisted('cnn.stage', 0)
  const s = Math.min(stageIdx, stages.length - 1)
  const stage = stages[s]
  const [chans, setChans] = usePersisted<Record<number, number>>('cnn.chan', {})
  const actOf = (st: ConvStage) => out[st.act?.runtime ?? st.conv.runtime ?? '']
  const [, oh, ow] = stage.conv.out_shape
  const c = Math.min(chans[s] ?? defaultChannel(actOf(stage), stage.conv.out_shape), stage.conv.out_shape[0] - 1)
  const setC = (v: number) => setChans((p) => ({ ...p, [s]: v }))

  // diagram click -> stage / classifier
  useEffect(() => {
    if (!arch.selected) return
    const i = stages.findIndex((st) => [st.conv.id, st.act?.id, st.pool?.id].includes(arch.selected!))
    if (i >= 0 && i !== s) setStageIdx(i)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arch.selected])
  useEffect(() => {
    if (![stage.conv.id, stage.act?.id, stage.pool?.id].includes(arch.selected ?? '')) arch.select(stage.conv.id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s])

  const preReached = (image && reached.has(image.id) ? 1 : 0) + 1 + (std && reached.has(std.id) ? 1 : 0)

  const spec3 = useMemo(() => cnnSpec(trace, loaded.params!, out, execution.pixels, classes), [trace, loaded.params, out, execution.pixels, classes])
  const summary3 = t('ex.three.summary', { layers: spec3.layers.map((l) => l.label).join(' → ') })
  const actKey = (stage.act ?? stage.conv).id
  const sel3 = useMemo<Pick3D | null>(() => {
    const li = spec3.layers.findIndex((l) => l.key === actKey)
    return li >= 0 ? { layer: li, index: c } : null
  }, [spec3, actKey, c])
  const pick3 = (p: Pick3D) => {
    const l = spec3.layers[p.layer]
    if (l.kind !== 'maps' || l.stage == null) return
    setStageIdx(l.stage)
    if (l.c > 1) setChans((prev) => ({ ...prev, [l.stage!]: p.index }))
  }

  return (
    <div className="ex-view ex-cnn">
      <div className="ex-view__bar">
        <PlaybackBar pb={pb} caption={caption} unit={t('ex.shell.stepsUnit')} />
        <p className="visually-hidden" aria-live="polite">
          {pb.playing ? '' : caption}
        </p>
      </div>

      <Panel id="ex-cnn-flow" title={t('ex.cnn.flowTitle')} className="ex-net">
        <NetSwitch
          three={(fail) => (
            <>
              <Net3D spec={spec3} revealed={reached} selected={sel3} onPick={pick3} label={summary3} onUnavailable={fail} />
              <ul className="ex-lod">
                {spec3.edges.map((e) => (
                  <li key={e.layerId} className="num">
                    {t('ex.three.edges', { a: spec3.layers[e.from].key, b: spec3.layers[e.to].key, k: e.list.length, n: int(e.total) })}
                  </li>
                ))}
              </ul>
            </>
          )}
          flat={() => <ForwardStrip loaded={loaded} execution={execution} reached={reached} stages={stages} chans={chans} stageIdx={s} onStage={setStageIdx} classes={classes} />}
        />
      </Panel>

      {std && (
        <Panel id="ex-cnn-pre" title={t('ex.cnn.preTitle')}>
          <PreprocessStrip pixels={execution.pixels} x={out[std.runtime ?? 'x']} mean={std.mean!} std={std.std!} eps={std.eps ?? 0} flatten={false} reached={preReached} side={side} />
        </Panel>
      )}

      <StagePanel key={s} stage={stage} stages={stages} s={s} setStage={setStageIdx} c={c} setC={setC} loaded={loaded} execution={execution} reached={reached} side={side} oh={oh} ow={ow} />

      <Classifier loaded={loaded} execution={execution} blocks={blocks} classes={classes} reached={reached} />

      <SoftmaxPanel logits={out[blocks[blocks.length - 1].zRuntime ?? 'logits']} probs={execution.probs} classes={classes} revealed={!soft || reached.has(soft.id)} />
    </div>
  )
}

/** The channel with the strongest activation: a sensible first thing to look at. */
function defaultChannel(act: Float32Array, s: number[]): number {
  const [C, H, W] = s
  let best = 0
  let bv = -Infinity
  for (let ch = 0; ch < C; ch++) {
    let m = -Infinity
    for (let i = 0; i < H * W; i++) m = Math.max(m, act[ch * H * W + i])
    if (m > bv) {
      bv = m
      best = ch
    }
  }
  return best
}

// --------------------------------------------------------------------------- overview of every layer
function ForwardStrip({
  loaded,
  execution,
  reached,
  stages,
  chans,
  stageIdx,
  onStage,
  classes,
}: {
  loaded: ExplorerViewProps['loaded']
  execution: ExplorerViewProps['execution']
  reached: Set<string>
  stages: ConvStage[]
  chans: Record<number, number>
  stageIdx: number
  onStage: (i: number) => void
  classes: string[]
}) {
  const { t } = useI18n()
  const { trace } = loaded
  const out = execution.outputs
  const stageOf = (id: string) => stages.findIndex((st) => [st.conv.id, st.act?.id, st.pool?.id].includes(id))
  return (
    <ol className="ex-flow">
      {trace.layers.map((l, i) => {
        const rt = l.op === 'input' ? null : runtimeOf(trace, l.id)
        const v = rt ? out[rt] : l.op === 'flatten' ? execution.pixels : null
        const si = stageOf(l.id)
        let viz: ReactNode = null
        let note = ''
        if (l.op === 'input') viz = <HeatMap values={execution.pixels} width={side(l)} height={side(l)} mode="gray" max={1} label={t('ex.shell.pre.image')} />
        else if (v && l.out_shape.length === 3) {
          const [C, H, W] = l.out_shape
          const ch = si >= 0 ? Math.min(C - 1, chans[si] ?? defaultChannel(out[stages[si].act?.runtime ?? stages[si].conv.runtime ?? ''], stages[si].conv.out_shape)) : 0
          viz = <HeatMap values={channel(v, ch, H, W)} width={W} height={H} mode={l.op === 'conv2d' ? 'diverging' : 'active'} max={absMax(v)} label={`${l.id} ${shape(l.out_shape)}`} />
          note = C > 1 ? t('ex.cnn.chanOf', { c: ch, n: C }) : ''
        } else if (v && l.out_shape.length === 2) viz = <HeatMap values={v} width={l.out_shape[1]} height={l.out_shape[0]} mode="diverging" label={l.id} />
        else if (v && l.out_shape.length === 1 && l.out_shape[0] <= 32) {
          const p = l.op === 'softmax' ? execution.probs : v
          const m = l.op === 'softmax' ? 1 : absMax(v)
          viz = (
            <span className="ex-flow__bars" aria-hidden="true">
              {Array.from(p, (x, k) => (
                <span key={k} data-neg={x < 0} style={{ '--h': `${(Math.abs(x) / m) * 100}%` } as CSSProperties} title={`${classes[k] ?? k}: ${fx(x, 3)}`} />
              ))}
            </span>
          )
        } else if (v && l.out_shape.length === 1) viz = <HeatMap values={v} width={Math.min(v.length, 100)} height={Math.ceil(v.length / 100)} mode="diverging" label={l.id} />
        const lit = reached.has(l.id)
        return (
          <li key={l.id} className="ex-flow__item" data-on={lit} data-sel={si === stageIdx}>
            <button type="button" className="ex-flow__card" onClick={() => si >= 0 && onStage(si)} disabled={si < 0} aria-label={`${i + 1}. ${t(`ex.arch.op.${l.op}` as Key)} ${shape(l.out_shape)}`}>
              <span className="ex-flow__op">{t(`ex.arch.op.${l.op}` as Key)}</span>
              <span className="ex-flow__viz">{viz}</span>
              <span className="num ex-flow__shape">{shape(l.out_shape)}</span>
              {note && <span className="ex-flow__note">{note}</span>}
            </button>
          </li>
        )
      })}
    </ol>
  )
}

const side = (l: { out_shape: number[] }) => l.out_shape[l.out_shape.length - 1]

// --------------------------------------------------------------------------- one conv stage
interface StageProps {
  stage: ConvStage
  stages: ConvStage[]
  s: number
  setStage: (i: number) => void
  c: number
  setC: (c: number) => void
  loaded: ExplorerViewProps['loaded']
  execution: ExplorerViewProps['execution']
  reached: Set<string>
  side: number
  oh: number
  ow: number
}

function StagePanel({ stage, stages, s, setStage, c, setC, loaded, execution, reached, side: imgSide, oh, ow }: StageProps) {
  const { t } = useI18n()
  const params = loaded.params!
  const out = execution.outputs
  const K = params.get(stage.kernel)
  const B = params.get(stage.bias)
  const [cout, cin, kh, kw] = stage.kernelShape
  const [, inH, inW] = stage.inShape
  const input = out[stage.inputRuntime ?? 'x']
  const conv = out[stage.conv.runtime ?? '']
  const act = stage.act?.runtime ? out[stage.act.runtime] : null
  const pool = stage.pool?.runtime ? out[stage.pool.runtime] : null
  const poolShape = stage.pool?.out_shape
  const N = oh * ow

  const [cinSel, setCinSel] = usePersisted<number | 'meanabs'>(`cnn.cin.${s}`, cin > 1 ? 'meanabs' : 0)
  const ci = cinSel === 'meanabs' ? null : Math.min(cinSel, cin - 1)
  const [locRaw, setLoc] = usePersisted<number | null>(`cnn.loc.${s}`, null)
  const defaultLoc = useMemo(() => channelArgmax(act ?? conv, c, oh, ow), [act, conv, c, oh, ow])
  const [following, setFollowing] = useState(false)

  // kernel sliding over the stage input: one step per output location, raster order
  const raw = usePlayback(N, `${execution.inputId}/${s}`, 30, false)
  useEffect(() => {
    if (following && raw.step >= 1) setLoc(raw.step - 1)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw.step, following])
  const slide: Playback = {
    ...raw,
    play: () => (setFollowing(true), raw.play()),
    toggle: () => (setFollowing(true), raw.toggle()),
    next: () => (setFollowing(true), raw.next()),
    prev: () => (setFollowing(true), raw.prev()),
    seek: (n: number) => (setFollowing(true), raw.seek(n)),
    replay: () => (setFollowing(true), raw.replay()),
  }
  const sliding = raw.step < N
  const loc = Math.min(N - 1, sliding && raw.step >= 1 ? raw.step - 1 : (locRaw ?? defaultLoc))
  const { r, q } = rasterPos(loc, ow)
  const computed = sliding ? raw.step : null

  const pick = (cell: number) => {
    setFollowing(false)
    setLoc(cell)
    raw.seek(N)
  }

  // the rebuilt computation at (c, r, q)
  const patch = useMemo(() => convPatch(input, stage.inShape, K, stage.kernelShape, B, c, r, q, stage.stride), [input, stage, K, B, c, r, q])
  const zGraph = conv[c * N + r * ow + q]
  const aGraph = act ? act[c * N + r * ow + q] : null
  const rf = receptiveField(stage.convStack, r, q)
  const pcell = stage.pool && poolShape ? poolCell(r, q, stage.pool.window!, stage.pool.stride ?? stage.pool.window!, poolShape[1], poolShape[2]) : null
  const pw = pcell && act && stage.pool ? poolWindow(act, stage.conv.out_shape, c, pcell.r, pcell.q, stage.pool.window!, stage.pool.stride ?? stage.pool.window!) : null
  const pGraph = pcell && pool && poolShape ? pool[c * poolShape[1] * poolShape[2] + pcell.r * poolShape[2] + pcell.q] : null
  const poolRf = pcell && stage.poolStack ? receptiveField(stage.poolStack, pcell.r, pcell.q) : null
  const ready = pool && poolShape && stage.pool ? pooledReady(computed ?? N, ow, stage.pool.window!, stage.pool.stride ?? stage.pool.window!, poolShape[1], poolShape[2]) : null

  const kMax = useMemo(() => {
    let m = 0
    for (let co = 0; co < cout; co++) m = Math.max(m, absMax(kernelSlice(K, stage.kernelShape, co, ci ?? 'meanabs')))
    return m
  }, [K, stage.kernelShape, cout, ci])
  const actMax = useMemo(() => absMax(act ?? conv), [act, conv])
  const convMax = useMemo(() => absMax(conv), [conv])
  const inCh = ci ?? (cin === 1 ? 0 : null)
  // input map drawn under the sliding window: the chosen input channel, or the mean over channels
  const inMap = useMemo(() => {
    if (inCh != null) return channel(input, inCh, inH, inW)
    const m = new Float32Array(inH * inW)
    for (let k = 0; k < cin; k++) for (let i = 0; i < inH * inW; i++) m[i] += input[k * inH * inW + i] / cin
    return m
  }, [input, inCh, cin, inH, inW])

  const lit = (id?: string) => !id || reached.has(id)
  const live = !raw.playing ? t('ex.cnn.slideLive', { r, q, c, z: fx(zGraph, 3) }) : ''
  const stageTitle = (st: ConvStage) => t('ex.cnn.stageTab', { k: st.index + 1, cin: st.kernelShape[1], cout: st.kernelShape[0], ks: `${st.kernelShape[2]}×${st.kernelShape[3]}` })

  return (
    <Panel id="ex-cnn-stage" title={t('ex.cnn.stageTitle', { k: s + 1, n: stages.length })} className="ex-stage">
      {stages.length > 1 && (
        <div className="ex-layerpick" role="group" aria-label={t('ex.cnn.pickStage')}>
          {stages.map((st, i) => (
            <button key={i} type="button" className="btn btn--quiet ex-chip" aria-pressed={i === s} onClick={() => setStage(i)}>
              {stageTitle(st)}
            </button>
          ))}
        </div>
      )}

      {/* kernels */}
      <div className="ex-block">
        <h5 className="ex-block__title">
          {t('ex.cnn.kernelsTitle', { n: cout, k: `${kh}×${kw}`, cin })} <span className="num ex-muted">{shape(stage.kernelShape)}</span>
        </h5>
        {cin > 1 && (
          <div className="ex-layerpick" role="group" aria-label={t('ex.cnn.pickCin')}>
            <button type="button" className="btn btn--quiet ex-chip" aria-pressed={cinSel === 'meanabs'} onClick={() => setCinSel('meanabs')}>
              {t('ex.cnn.meanAbs')}
            </button>
            {Array.from({ length: cin }, (_, k) => (
              <button key={k} type="button" className="btn btn--quiet ex-chip" aria-pressed={cinSel === k} onClick={() => setCinSel(k)}>
                {t('ex.cnn.inCh', { c: k })}
              </button>
            ))}
          </div>
        )}
        <p className="ex-hint">{ci == null && cin > 1 ? t('ex.cnn.meanAbsNote', { cin }) : t('ex.cnn.kernelNote', { c: ci ?? 0 })}</p>
        <div className="ex-thumbs ex-thumbs--kernels" role="listbox" aria-label={t('ex.cnn.kernelsAria', { n: cout })}>
          {Array.from({ length: cout }, (_, co) => (
            <button key={co} type="button" role="option" aria-selected={co === c} className="ex-thumb ex-thumb--btn" onClick={() => setC(co)}>
              <HeatMap values={kernelSlice(K, stage.kernelShape, co, ci ?? 'meanabs')} width={kw} height={kh} mode={ci == null && cin > 1 ? 'active' : 'diverging'} max={kMax} label={t('ex.cnn.kernelAria', { c: co })} />
              <span className="num ex-thumb__cap">
                <span>{t('ex.cnn.kShort', { c: co })}</span>
                <span className="ex-muted">b {fx(B[co], 2, true)}</span>
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* channel thumbnails */}
      <div className="ex-block">
        <h5 className="ex-block__title">
          {t('ex.cnn.mapsTitle', { op: t(stage.act ? 'ex.arch.op.relu' : 'ex.arch.op.conv2d') })} <span className="num ex-muted">{shape(stage.conv.out_shape)}</span>
        </h5>
        <div className="ex-thumbs ex-thumbs--maps" role="listbox" aria-label={t('ex.cnn.mapsAria', { n: cout })}>
          {Array.from({ length: cout }, (_, co) => (
            <button key={co} type="button" role="option" aria-selected={co === c} className="ex-thumb ex-thumb--btn" onClick={() => setC(co)} data-on={lit(stage.act?.id ?? stage.conv.id)}>
              <HeatMap values={channel(act ?? conv, co, oh, ow)} width={ow} height={oh} mode={act ? 'active' : 'diverging'} max={actMax} label={t('ex.cnn.chanOf', { c: co, n: cout })} />
              <span className="num">{t('ex.cnn.chShort', { c: co })}</span>
            </button>
          ))}
        </div>
      </div>

      {/* sliding window */}
      <div className="ex-block">
        <h5 className="ex-block__title">{t('ex.cnn.slideTitle', { c, n: cout })}</h5>
        <PlaybackBar pb={slide} caption={t('ex.cnn.slideCaption', { r, q, n: N, k: `${kh}×${kw}` })} unit={t('ex.cnn.positions')} />
        <p className="visually-hidden" aria-live="polite">
          {live}
        </p>
        <div className="ex-maps">
          <figure className="ex-maps__fig">
            <HeatMap values={execution.pixels} width={imgSide} height={imgSide} mode="gray" max={1} label={t('ex.cnn.rfAria', { y0: rf.y0, y1: rf.y1 - 1, x0: rf.x0, x1: rf.x1 - 1 })} box={{ x0: rf.x0, y0: rf.y0, x1: rf.x1, y1: rf.y1 }} />
            <figcaption>
              <b>{t('ex.cnn.rfTitle')}</b>
              <span className="num">{t('ex.cnn.rfRange', { y0: rf.y0, y1: rf.y1 - 1, x0: rf.x0, x1: rf.x1 - 1, h: rf.y1 - rf.y0, w: rf.x1 - rf.x0 })}</span>
            </figcaption>
          </figure>
          <figure className="ex-maps__fig">
            <HeatMap
              values={inMap}
              width={inW}
              height={inH}
              mode="diverging"
              label={t('ex.cnn.inAria')}
              box={{ x0: patch.x0, y0: patch.y0, x1: patch.x0 + kw, y1: patch.y0 + kh }}
            />
            <figcaption>
              <b>{s === 0 ? t('ex.cnn.inX') : t('ex.cnn.inPrev', { id: stage.inputId })}</b>
              <span className="num">
                {shape(stage.inShape)}
                {cin > 1 ? ` · ${inCh == null ? t('ex.cnn.meanOver', { cin }) : t('ex.cnn.inCh', { c: inCh })}` : ''}
              </span>
            </figcaption>
          </figure>
          <figure className="ex-maps__fig" data-on={lit(stage.conv.id)}>
            <HeatMap values={reveal(channel(conv, c, oh, ow), computed)} width={ow} height={oh} mode="diverging" max={convMax} label={t('ex.cnn.convAria', { c })} selected={loc} onSelect={pick} />
            <figcaption>
              <b>{t('ex.cnn.convMap', { c })}</b>
              <span className="num">{shape([oh, ow])}</span>
            </figcaption>
          </figure>
          {act && (
            <figure className="ex-maps__fig" data-on={lit(stage.act?.id)}>
              <HeatMap values={reveal(channel(act, c, oh, ow), computed)} width={ow} height={oh} mode="active" max={actMax} label={t('ex.cnn.actAria', { c })} selected={loc} onSelect={pick} />
              <figcaption>
                <b>{t('ex.cnn.actMap', { c })}</b>
                <span className="num">{shape([oh, ow])}</span>
              </figcaption>
            </figure>
          )}
          {pool && poolShape && (
            <figure className="ex-maps__fig" data-on={lit(stage.pool?.id)}>
              <HeatMap
                values={(() => {
                  const ch = channel(pool, c, poolShape[1], poolShape[2])
                  if (!ready || computed == null) return ch
                  return Float32Array.from(ch, (v, i) => (ready[i] ? v : 0))
                })()}
                width={poolShape[2]}
                height={poolShape[1]}
                mode="active"
                max={actMax}
                label={t('ex.cnn.poolAria', { c })}
                selected={pcell ? pcell.r * poolShape[2] + pcell.q : null}
                onSelect={(cell) => {
                  const pr = Math.floor(cell / poolShape[2])
                  const pq = cell % poolShape[2]
                  const w = poolWindow(act!, stage.conv.out_shape, c, pr, pq, stage.pool!.window!, stage.pool!.stride ?? stage.pool!.window!)
                  pick((w.y0 + w.dy) * ow + (w.x0 + w.dx))
                }}
              />
              <figcaption>
                <b>{t('ex.cnn.poolMap', { c })}</b>
                <span className="num">{shape(poolShape.slice(1))}</span>
              </figcaption>
            </figure>
          )}
        </div>
        <Hint>{t('ex.cnn.mapsHonest')}</Hint>
      </div>

      {/* inspector of (c, r, q) */}
      <div className="ex-block ex-loc" aria-label={t('ex.cnn.locTitle', { c, r, q })}>
        <h5 className="ex-block__title">{t('ex.cnn.locTitle', { c, r, q })}</h5>
        <ConvProducts patch={patch} cin={cin} kh={kh} kw={kw} ci={ci} onCin={(k) => setCinSel(k)} />
        <dl className="ex-terms">
          <Term label={t('ex.cnn.sumProducts', { n: cin * kh * kw })} value={patch.sum} />
          <Term label={t('ex.cnn.biasOf', { c })} value={patch.bias} />
        </dl>
        <CheckRow label={<span className="num">z = Σ + b</span>} rebuilt={patch.z} runtime={zGraph} />
        {aGraph != null && <CheckRow label={<span className="num">a = max(0, z)</span>} rebuilt={Math.max(0, patch.z)} runtime={aGraph} />}
        {stage.pool &&
          (pw && pcell && pGraph != null ? (
            <div className="ex-poolwin">
              <p className="ex-dense__cap">
                {t('ex.cnn.poolWin', { k: `${pw.size}×${pw.size}`, pr: pcell.r, pq: pcell.q, y0: pw.y0, x0: pw.x0 })}
                {poolRf && <span className="num ex-muted"> · {t('ex.cnn.poolRf', { h: poolRf.y1 - poolRf.y0, w: poolRf.x1 - poolRf.x0 })}</span>}
              </p>
              <div className="ex-grid" style={{ '--k': pw.size } as CSSProperties} role="group" aria-label={t('ex.cnn.poolWinAria')}>
                {Array.from(pw.values, (v, i) => (
                  <span key={i} className="num ex-grid__cell" data-max={i === pw.argmax} data-here={pw.y0 + Math.floor(i / pw.size) === r && pw.x0 + (i % pw.size) === q}>
                    {fx(v, 3)}
                    {i === pw.argmax && <span className="visually-hidden"> {t('ex.cnn.isMax')}</span>}
                  </span>
                ))}
              </div>
              <CheckRow label={<span className="num">max</span>} rebuilt={pw.max} runtime={pGraph} />
              <p className="ex-hint">{pw.y0 + pw.dy === r && pw.x0 + pw.dx === q ? t('ex.cnn.poolPassed') : t('ex.cnn.poolBlocked', { dy: pw.y0 + pw.dy, dx: pw.x0 + pw.dx })}</p>
            </div>
          ) : (
            <p className="ex-hint">{t('ex.cnn.poolDropped')}</p>
          ))}
      </div>
    </Panel>
  )
}

/** patch × kernel = products, as three small k × k number grids for one input channel, plus every channel's partial sum. */
function ConvProducts({ patch, cin, kh, kw, ci, onCin }: { patch: ReturnType<typeof convPatch>; cin: number; kh: number; kw: number; ci: number | null; onCin: (c: number) => void }) {
  const { t } = useI18n()
  const n = kh * kw
  const partial = Array.from({ length: cin }, (_, k) => {
    let s = 0
    for (let i = 0; i < n; i++) s += patch.products[k * n + i]
    return s
  })
  const shown = ci ?? partial.reduce((b, v, k) => (Math.abs(v) > Math.abs(partial[b]) ? k : b), 0)
  const slice = (a: Float64Array) => a.subarray(shown * n, (shown + 1) * n)
  const pm = absMax(slice(patch.products))
  const grid = (a: Float64Array, label: string, tint?: boolean) => (
    <figure className="ex-prod__fig">
      <div className="ex-grid" style={{ '--k': kw } as CSSProperties} role="group" aria-label={label}>
        {Array.from(a, (v, i) => (
          <span key={i} className="num ex-grid__cell" data-sign={tint ? (v > 0 ? 'pos' : v < 0 ? 'neg' : 'zero') : undefined} style={tint ? ({ '--a': Math.abs(v) / pm } as CSSProperties) : undefined}>
            {fx(v, 2)}
          </span>
        ))}
      </div>
      <figcaption>{label}</figcaption>
    </figure>
  )
  return (
    <div className="ex-prod">
      {cin > 1 && <p className="ex-dense__cap">{t('ex.cnn.prodChannel', { c: shown, cin })}</p>}
      <div className="ex-prod__row">
        {grid(slice(patch.patch), t('ex.cnn.patch'))}
        <span className="ex-prod__op" aria-hidden="true">
          ×
        </span>
        {grid(slice(patch.kernel), t('ex.cnn.weights'))}
        <span className="ex-prod__op" aria-hidden="true">
          =
        </span>
        {grid(slice(patch.products), t('ex.cnn.products'), true)}
      </div>
      {cin > 1 && (
        <div className="ex-partials" role="group" aria-label={t('ex.cnn.partials')}>
          <p className="ex-dense__cap">{t('ex.cnn.partials')}</p>
          <div className="ex-partials__list">
            {partial.map((v, k) => (
              <button key={k} type="button" className="ex-partial" aria-pressed={k === shown} onClick={() => onCin(k)}>
                <span className="num">{t('ex.cnn.inCh', { c: k })}</span>
                <b className="num" data-sign={v >= 0 ? 'pos' : 'neg'}>
                  {fx(v, 3, true)}
                </b>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------- flatten + classifier
function Classifier({
  loaded,
  execution,
  blocks,
  classes,
  reached,
}: {
  loaded: ExplorerViewProps['loaded']
  execution: ExplorerViewProps['execution']
  blocks: ReturnType<typeof denseBlocks>
  classes: string[]
  reached: Set<string>
}) {
  const { t } = useI18n()
  const { trace } = loaded
  const params = loaded.params!
  const out = execution.outputs
  const [bi, setBi] = usePersisted('cnn.block', blocks.length - 1)
  const b = blocks[Math.min(bi, blocks.length - 1)]
  const values = out[b.aRuntime ?? b.zRuntime ?? '']
  const [jSel, setJ] = usePersisted<number | null>(`cnn.j.${b.index}`, null)
  const j = jSel != null && jSel < b.nOut ? jSel : argmax(values)
  const [picked, setPicked] = usePersisted<number | null>('cnn.picked', null)
  const input = out[b.inputRuntime ?? '']
  const byId = new Map(trace.layers.map((l) => [l.id, l]))
  const inLayer = byId.get(b.inputId)
  const src = inLayer?.op === 'flatten' ? byId.get(inLayer.inputs[0]) : null
  const spatial: SpatialInput = src && src.out_shape.length === 3 ? { kind: 'maps', c: src.out_shape[0], h: src.out_shape[1], w: src.out_shape[2] } : null
  const name = (i: number) => (spatial?.kind === 'maps' ? t('ex.cnn.flatName', { i, c: Math.floor(i / (spatial.h * spatial.w)), r: Math.floor((i % (spatial.h * spatial.w)) / spatial.w), q: i % spatial.w }) : `#${i}`)
  return (
    <Panel id="ex-cnn-head" title={t('ex.cnn.headTitle')} sub={inLayer?.op === 'flatten' && src ? t('ex.cnn.flatSub', { s: shape(src.out_shape), n: int(b.nIn) }) : undefined}>
      {blocks.length > 1 && (
        <div className="ex-layerpick" role="group" aria-label={t('ex.mlp.pickLayer')}>
          {blocks.map((x, i) => (
            <button key={i} type="button" className="btn btn--quiet ex-chip" aria-pressed={i === b.index} onClick={() => setBi(i)}>
              {x.dense.id} <span className="num">{`${x.nIn}→${x.nOut}`}</span>
            </button>
          ))}
        </div>
      )}
      <div className="ex-flatstrip" data-on={reached.has(b.inputId)}>
        <HeatMap values={input} width={Math.min(input.length, 100)} height={Math.ceil(input.length / 100)} mode="active" label={t('ex.cnn.flatAria', { n: input.length })} selected={picked != null && picked < input.length ? picked : null} onSelect={(i) => setPicked(i)} />
        <p className="ex-hint num">{t('ex.cnn.flatNote', { n: int(input.length), w: Math.min(input.length, 100) })}</p>
      </div>
      {b.isOutput && (
        <div className="ex-layerpick" role="group" aria-label={t('ex.cnn.pickClass')}>
          {classes.map((cl, k) => (
            <button key={k} type="button" className="btn btn--quiet ex-chip" aria-pressed={k === j} onClick={() => setJ(k)}>
              {cl}
            </button>
          ))}
        </div>
      )}
      <h5 className="ex-block__title">{b.isOutput ? t('ex.mlp.logitTitle', { k: classes[j] ?? j }) : t('ex.cnn.unitTitle', { j, id: b.dense.id })}</h5>
      <DenseInspector
        input={input}
        sym="a"
        W={params.get(b.weight)}
        b={params.get(b.bias)}
        nIn={b.nIn}
        nOut={b.nOut}
        j={j}
        zGraph={out[b.zRuntime ?? ''][j]}
        aGraph={b.aRuntime ? out[b.aRuntime][j] : null}
        spatial={spatial}
        picked={picked}
        onPick={setPicked}
        sourceName={name}
      />
    </Panel>
  )
}
