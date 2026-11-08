import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { useI18n } from '../../i18n'
import { usePlayback } from '../../engine/react'
import type { AdaBoostEnsemble } from '../../engine/adaboost'
import { bytes, int } from '../../lib/format'
import { Segmented } from '../Segmented'
import { HeatMap } from './HeatMap'
import { PlaybackBar } from './PlaybackBar'
import type { ExplorerViewProps } from './types'
import { LineChart } from './classic/LineChart'
import { extent, fmt, leaderChanges, routeUsage, rowCol, shareSeries, splitUsagePrefix, treeSplitCounts } from './classic/model'
import { Legend, MapFigure, Narration } from './classic/parts'
import { TreeDiagram } from './classic/TreeDiagram'
import { imageShape, layerOf } from './classic/trace'
import { useWidth } from './classic/useCssColors'

/** Split counts over the whole ensemble, computed once per parsed ensemble. */
const usageCache = new WeakMap<AdaBoostEnsemble, Int32Array>()

const pctText = (v: number) => `${(v * 100).toFixed(v >= 0.9995 || v === 0 ? 0 : 1)}%`

/**
 * AdaBoost-SAMME, round by round: the tree of the selected round with the route the input takes,
 * the weighted vote it casts, the running vote shares of all classes, and where the trees look.
 * Vote shares are alpha-weighted votes divided by their sum: they are not calibrated probabilities.
 */
export default function AdaBoostView({ classes, loaded, execution }: ExplorerViewProps) {
  const { t } = useI18n()
  const ens = loaded.ensemble!
  const run = execution.ada
  const layer = layerOf(loaded.trace, 'tree_ensemble')
  const { h, w } = imageShape(loaded.trace)
  const N = layer.n_estimators ?? ens.trees.length
  const K = classes.length
  const nPix = h * w
  const pb = usePlayback(N, `${execution.seq}`, 40)
  const step = pb.step
  const ti = Math.max(1, step) - 1
  const [hlState, setHl] = useState(execution.top)
  const hl = Math.min(hlState, K - 1)
  const [usageMode, setUsageMode] = useState<'route' | 'all'>('route')
  const [allUsage, setAllUsage] = useState<Int32Array | null>(() => usageCache.get(ens) ?? null)
  const treeBox = useRef<HTMLDivElement>(null)
  const boxW = useWidth(treeBox)

  useEffect(() => {
    if (usageMode !== 'all' || allUsage) return
    const id = window.setTimeout(() => {
      const u = usageCache.get(ens) ?? splitUsagePrefix(ens, nPix)
      usageCache.set(ens, u)
      setAllUsage(u)
    }, 30)
    return () => window.clearTimeout(id)
  }, [usageMode, allUsage, ens, nPix])

  const shares = useMemo(() => (run ? shareSeries(run.cumulative, K) : new Float64Array((N + 1) * K)), [run, K, N])
  const changes = useMemo(() => leaderChanges(shares, K, true), [shares, K])
  const facts = useMemo(() => {
    const counts = layer.node_counts ?? ens.nodeCounts
    const depths = layer.depths ?? ens.depths
    const [nlo, nhi] = extent(counts)
    const [dlo, dhi] = extent(depths)
    return { nlo, nhi, dlo, dhi, total: counts.reduce((s, v) => s + v, 0) }
  }, [layer, ens])
  const treeCounts = useMemo(() => treeSplitCounts(ens.trees[ti], nPix), [ens, ti, nPix])
  const distinct = useMemo(() => treeCounts.reduce((s, v) => s + (v > 0 ? 1 : 0), 0), [treeCounts])
  const routeU = useMemo(() => (run ? routeUsage(run, step, nPix) : new Float32Array(nPix)), [run, step, nPix])
  const allU = useMemo(() => (allUsage ? allUsage.subarray(step * nPix, (step + 1) * nPix) : null), [allUsage, step, nPix])

  if (!run) return <p className="msg msg--error">{t('ex.ada.noRun')}</p>

  const round = run.rounds[ti]
  const leader = step > 0 ? run.leader[step] : -1
  const cum = run.cumulative[step]
  const shareNow = shares.subarray(step * K, (step + 1) * K)
  const alphaSum = cum.reduce((s, v) => s + v, 0)
  const routePixels = [...new Set(round.path.map((p) => p.feature))]
  const animate = !pb.playing && !pb.reduced
  const fullDepth = boxW > 0 && boxW < 700 ? 1 : 2
  const lastChange = changes.filter((c) => c.step <= step).pop()
  const narr = pb.playing
    ? ''
    : step === 0
      ? t('ex.ada.narrStart', { n: N })
      : t('ex.ada.narr', { t: step, n: N, name: classes[round.klass], a: fmt(round.alpha, 3), leader: classes[leader], s: pctText(shareNow[leader]) })
  const caption =
    step === 0 ? t('ex.ada.capStart') : t('ex.ada.cap', { t: step, n: N, name: classes[round.klass], leader: classes[leader] })
  const shownChanges = changes.filter((c) => c.step > 0)

  return (
    <div className="exc">
      <p className="exc-intro">{t('ex.ada.intro')}</p>

      <dl className="exc-facts">
        <div>
          <dt>{t('ex.ada.factRounds')}</dt>
          <dd className="num">{int(N)}</dd>
        </div>
        <div>
          <dt>{t('ex.ada.factDepth')}</dt>
          <dd className="num">{facts.dlo === facts.dhi ? facts.dlo : `${facts.dlo}–${facts.dhi}`}</dd>
        </div>
        <div>
          <dt>{t('ex.ada.factNodes')}</dt>
          <dd className="num">
            {int(facts.nlo)}–{int(facts.nhi)}
          </dd>
        </div>
        <div>
          <dt>{t('ex.ada.factTotal')}</dt>
          <dd className="num">{int(facts.total)}</dd>
        </div>
        <div>
          <dt>{t('ex.ada.factFile')}</dt>
          <dd className="num">{bytes(loaded.entry.size_bytes)}</dd>
        </div>
      </dl>
      <p className="note note--muted">{t('ex.ada.parsedNote', { size: bytes(loaded.entry.size_bytes), ms: Math.round(loaded.loadMs) })}</p>

      <section className="exc-sec" aria-label={t('ex.ada.roundsTitle')}>
        <PlaybackBar pb={pb} caption={caption} unit={t('ex.ada.unitRounds')} />
      </section>

      <div className="exc-grid exc-grid--ada">
        <section className="exc-sec" aria-label={t('ex.ada.treeTitle', { t: ti + 1 })} ref={treeBox}>
          <h4 className="exc-h">
            {t('ex.ada.treeTitle', { t: ti + 1 })}
            {step === 0 && <span className="exc-tag exc-tag--mute">{t('ex.ada.notYet')}</span>}
          </h4>
          <TreeDiagram ens={ens} t={ti} round={round} classes={classes} width={w} fullDepth={fullDepth} animate={animate} />
          <Legend
            items={[
              { swatch: 'accent', text: t('ex.ada.legRoute') },
              { swatch: 'mute', text: t('ex.ada.legStub') },
            ]}
          />
          <p className="note note--muted">{t('ex.ada.ruleNote')}</p>
        </section>

        <section className="exc-sec" aria-label={t('ex.ada.routeTitle')}>
          <h4 className="exc-h">{t('ex.ada.routeTitle')}</h4>
          <ol key={`${execution.seq}-${ti}`} className={animate ? 'exc-route is-anim' : 'exc-route'}>
            {round.path.map((s, i) => {
              const { r, c } = rowCol(s.feature, w)
              return (
                <li key={i} className="exc-route__step" style={{ '--d': i } as CSSProperties}>
                  <span className="exc-mini" aria-hidden="true">
                    <HeatMap values={execution.pixels} width={w} height={h} mode="gray" max={1} selected={s.feature} label="" />
                  </span>
                  <div className="exc-route__text">
                    <p className="exc-route__q">
                      <span className="exc-route__no num">{i + 1}</span>
                      {t('ex.ada.stepPixel', { r, c })}
                    </p>
                    <p className="num exc-route__a">
                      {t(s.goLeft ? 'ex.ada.stepLess' : 'ex.ada.stepMore', { v: fmt(s.pixel, 3), th: fmt(s.thresholdPixel, 3) })}
                    </p>
                    <p className="exc-route__skip num">{t('ex.ada.stepSkip', { n: int(s.otherSize) })}</p>
                  </div>
                </li>
              )
            })}
            <li className="exc-route__leaf" style={{ '--d': round.path.length } as CSSProperties}>
              <span className="exc-route__vote">{classes[round.klass]}</span>
              <div className="exc-route__text">
                <p className="exc-route__q">{t('ex.ada.leaf', { name: classes[round.klass] })}</p>
                <p className="num exc-route__a">{t('ex.ada.alpha', { a: fmt(round.alpha, 4), t: ti + 1 })}</p>
              </div>
            </li>
          </ol>
          <p className="note note--muted">{t('ex.ada.unitNote')}</p>
        </section>
      </div>

      <section className="exc-sec" aria-label={t('ex.ada.votesTitle')}>
        <h4 className="exc-h">{t('ex.ada.votesTitle')}</h4>
        <div className="exc-acc__body">
          <div className="exc-acc__chart">
            <LineChart
              rows={shares}
              K={K}
              steps={N}
              cursor={step}
              names={classes}
              highlight={hl}
              leader={leader}
              future="faint"
              marks={shownChanges.map((c) => c.step)}
              yFormat={(v) => `${Math.round(v * 100)}%`}
              domain={[0, Math.max(0.05, extent(shares)[1])]}
              height={220}
              label={t('ex.ada.chartAria')}
              valueText={caption}
              onSeek={pb.seek}
            />
            <p className="exc-fig__cap">
              {t('ex.ada.chartCap', { name: classes[hl] })}{' '}
              {lastChange ? t('ex.ada.lastChange', { t: lastChange.step, name: classes[lastChange.klass] }) : ''}
            </p>
            <p className="exc-changes num">
              {t('ex.ada.changes', { n: shownChanges.length })}{' '}
              {shownChanges
                .slice(0, 12)
                .map((c) => t('ex.ada.changeItem', { t: c.step, name: classes[c.klass] }))
                .join(' · ')}
              {shownChanges.length > 12 ? ` · ${t('ex.ada.more', { n: shownChanges.length - 12 })}` : ''}
            </p>
          </div>
          <div className="exc-acc__table">
            <table className="exc-scores">
              <caption>{t('ex.ada.barsCap', { t: step, n: N })}</caption>
              <thead>
                <tr>
                  <th scope="col">{t('ex.ada.class')}</th>
                  <th scope="col">Σα</th>
                  <th scope="col">{t('ex.ada.share')}</th>
                </tr>
              </thead>
              <tbody>
                {classes.map((name, c) => (
                  <tr key={c} className={c === leader ? 'is-lead' : c === hl ? 'is-sel' : undefined}>
                    <th scope="row">
                      <button type="button" className="exc-rowbtn" aria-pressed={c === hl} onClick={() => setHl(c)} title={t('ex.ada.highlight', { name })}>
                        {name}
                      </button>
                      {c === leader && <span className="exc-tag">{t('ex.classic.leader')}</span>}
                    </th>
                    <td className="num">{fmt(cum[c], 3)}</td>
                    <td>
                      <span className="exc-bar" aria-hidden="true">
                        <span className="exc-bar__fill" style={{ transform: `scaleX(${shareNow[c]})` }} />
                      </span>
                      <span className="num exc-bar__v">{pctText(shareNow[c])}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="note note--muted num">{t('ex.ada.sumNote', { s: fmt(alphaSum, 3) })}</p>
          </div>
        </div>
        <p className="note exc-honest">{t('ex.ada.honest')}</p>
      </section>

      <section className="exc-sec" aria-label={t('ex.ada.pixTitle')}>
        <h4 className="exc-h">{t('ex.ada.pixTitle')}</h4>
        <div className="exc-maps__row exc-maps__row--pix">
          <MapFigure title={t('ex.ada.pixTree', { t: ti + 1 })} caption={t('ex.ada.pixTreeCap', { n: distinct, s: int(treeCounts.reduce((a, b) => a + b, 0)), r: routePixels.length })}>
            <HeatMap values={treeCounts} width={w} height={h} mode="active" marks={routePixels} label={t('ex.ada.pixTreeAria', { t: ti + 1 })} />
          </MapFigure>
          <MapFigure
            title={usageMode === 'route' ? t('ex.ada.pixRoute', { t: step }) : t('ex.ada.pixAll', { t: step })}
            caption={usageMode === 'route' ? t('ex.ada.pixRouteCap') : t('ex.ada.pixAllCap')}
          >
            {usageMode === 'route' || allU ? (
              <HeatMap values={usageMode === 'route' ? routeU : allU!} width={w} height={h} mode="active" label={t('ex.ada.pixUseAria', { t: step })} />
            ) : (
              <div className="exc-pending" role="status">
                {t('ex.ada.indexing', { n: int(facts.total) })}
              </div>
            )}
          </MapFigure>
          <div className="exc-maps__ctl">
            <Segmented
              legend={t('ex.ada.pixMode')}
              value={usageMode}
              onChange={setUsageMode}
              options={[
                { value: 'route', label: t('ex.ada.pixModeRoute') },
                { value: 'all', label: t('ex.ada.pixModeAll') },
              ]}
            />
            <Legend items={[{ swatch: 'pos', text: t('ex.ada.legCount') }, { swatch: 'mark', text: t('ex.ada.legRoutePix') }]} />
          </div>
        </div>
      </section>
      <Narration text={narr} />
    </div>
  )
}
