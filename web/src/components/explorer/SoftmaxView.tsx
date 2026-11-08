import { useMemo, useState } from 'react'
import { useI18n } from '../../i18n'
import { softmaxClassTerms } from '../../engine/reconstruct'
import { HeatMap } from './HeatMap'
import type { ExplorerViewProps } from './types'
import { Accumulation, useAccumulation } from './classic/Accumulation'
import { absQuantile, classRow, extent, fmt, fmtSigned, makeAdditive, rowCol, rowOf } from './classic/model'
import { ClassPicker, Legend, MapFigure, Narration, Readout, VerifyTable } from './classic/parts'
import { imageShape, layerOf, runtimeOf } from './classic/trace'

/**
 * Softmax regression, computed in the open: the learned weight map and bias of every class, the
 * products x_i * W_ik for the current input, their accumulation into the logits, softmax, and a check
 * against the logits returned by the deployed graph.
 */
export default function SoftmaxView({ classes, loaded, execution }: ExplorerViewProps) {
  const { t } = useI18n()
  const { trace } = loaded
  const layer = layerOf(trace, 'dense')
  const { h, w } = imageShape(trace)
  const params = loaded.params!
  const W = params.get(layer.params!.weight)
  const b = params.get(layer.params!.bias)
  const N = layer.in_features!
  const K = layer.out_features!
  const x = runtimeOf(trace, execution, 'standardized_input')
  const graphLogits = runtimeOf(trace, execution, 'logits')

  const [kState, setK] = useState(execution.top)
  const k = Math.min(kState, K - 1)
  const [picked, setPicked] = useState<number | null>(null)
  const [hover, setHover] = useState<number | null>(null)

  // W is [N, K] row-major: the weight map of class q is the column W[:, q]
  const columns = useMemo(() => Array.from({ length: K }, (_, q) => Float32Array.from({ length: N }, (_, i) => W[i * K + q])), [W, N, K])
  const wMax = useMemo(() => {
    const [lo, hi] = extent(W)
    return Math.max(Math.abs(lo), Math.abs(hi))
  }, [W])
  const xMax = useMemo(() => {
    const [lo, hi] = extent(x)
    return Math.max(Math.abs(lo), Math.abs(hi))
  }, [x])
  const a = useMemo(
    () =>
      makeAdditive(K, N, (q) => {
        const s = softmaxClassTerms(x, W, b, N, K, q)
        return { terms: s.terms, start: s.bias }
      }),
    [K, N, x, W, b],
  )
  const acc = useAccumulation(a, k, `${execution.seq}`)
  const step = acc.pb.step
  const row = classRow(a, k)
  const termMax = useMemo(() => absQuantile(row, 0.99), [row])
  const applied = useMemo(() => {
    const out = new Float64Array(N)
    for (let i = 0; i < step; i++) out[acc.order[i]] = row[acc.order[i]]
    return out
  }, [N, step, acc.order, row])

  const j = hover ?? picked ?? acc.order[0]
  const { r, c } = rowCol(j, w)
  const term = row[j]
  const rank = acc.rank[j] + 1
  const scoreAt = rowOf(acc.running, K, step)
  const mapProps = { width: w, height: h, selected: j, onSelect: (p: number) => setPicked(p), onHover: setHover }
  const narr = acc.pb.playing
    ? ''
    : step === 0
      ? t('ex.sm.narrStart')
      : t('ex.sm.narr', { s: step, n: N, r: rowCol(acc.current!, w).r, c: rowCol(acc.current!, w).c, leader: classes[acc.leader], v: fmt(scoreAt[acc.leader], 2) })

  return (
    <div className="exc">
      <p className="exc-intro">{t('ex.sm.intro')}</p>

      <section className="exc-sec" aria-label={t('ex.sm.classTitle')}>
        <h4 className="exc-h">{t('ex.sm.classTitle')}</h4>
        <ClassPicker
          classes={classes}
          value={k}
          onChange={setK}
          width={w}
          height={h}
          label={t('ex.sm.classTitle')}
          thumb={(q) => ({ values: columns[q], mode: 'diverging', max: wMax })}
          sub={(q) => fmt(b[q], 3)}
        />
      </section>

      <div className="exc-grid">
        <section className="exc-maps" aria-label={t('ex.sm.mapsTitle', { name: classes[k] })}>
          <h4 className="exc-h">{t('ex.sm.mapsTitle', { name: classes[k] })}</h4>
          <div className="exc-maps__row exc-maps__row--3">
            <MapFigure title={t('ex.sm.mapX')} caption={t('ex.sm.mapXCap')}>
              <HeatMap {...mapProps} values={x} mode="diverging" max={xMax} label={t('ex.sm.mapXAria')} />
            </MapFigure>
            <MapFigure title={t('ex.sm.mapW', { name: classes[k] })} caption={t('ex.sm.mapWCap', { m: fmt(wMax, 3) })}>
              <HeatMap {...mapProps} values={columns[k]} mode="diverging" max={wMax} label={t('ex.sm.mapWAria', { name: classes[k] })} />
            </MapFigure>
            <MapFigure title={t('ex.sm.mapTerm', { name: classes[k] })} caption={t('ex.sm.mapTermCap', { s: step, n: N, m: fmt(termMax, 3) })}>
              <HeatMap
                {...mapProps}
                values={applied}
                mode="diverging"
                max={termMax}
                box={acc.current != null && step < N ? { x0: acc.current % w, y0: Math.floor(acc.current / w), x1: (acc.current % w) + 1, y1: Math.floor(acc.current / w) + 1 } : null}
                label={t('ex.sm.mapTermAria', { name: classes[k] })}
              />
            </MapFigure>
          </div>
          <Legend
            items={[
              { swatch: 'pos', text: t('ex.classic.legPos') },
              { swatch: 'neg', text: t('ex.classic.legNeg') },
              { swatch: 'mark', text: t('ex.classic.legSel') },
            ]}
          />
          <p className="note note--muted">{t('ex.classic.mapHint')}</p>
        </section>

        <section className="exc-inspect" aria-label={t('ex.sm.inspectTitle')}>
          <h4 className="exc-h">{t('ex.classic.pixelTitle', { r, c, name: classes[k] })}</h4>
          <Readout
            rows={[
              { sym: 'p', label: t('ex.classic.rawPixel'), value: fmt(execution.pixels[j], 3) },
              { sym: 'x', label: t('ex.classic.stdPixel'), value: fmt(x[j], 4) },
              { sym: 'W', label: t('ex.sm.weight'), value: fmt(W[j * K + k], 4) },
              { sym: 'x·W', label: term > 0 ? t('ex.sm.termFor') : term < 0 ? t('ex.sm.termAgainst') : t('ex.sm.termZero'), value: fmtSigned(term, 4), strong: true },
              { sym: 'b', label: t('ex.sm.bias'), value: fmt(b[k], 4) },
              { sym: 'z(s)', label: t('ex.sm.partial', { s: step }), value: fmt(scoreAt[k], 4) },
              { sym: 'z', label: t('ex.sm.logit'), value: fmt(a.total[k], 4) },
            ]}
          />
          <p className="note">{rank <= step ? t('ex.classic.applied', { s: rank }) : t('ex.classic.notApplied', { s: rank })}</p>
          <p className="note note--muted">{t('ex.sm.inspectNote')}</p>
        </section>
      </div>

      <Accumulation acc={acc} a={a} classes={classes} k={k} width={w} finalProbs={execution.probs} scoreName={t('ex.sm.score')} startName={t('ex.sm.start')} />

      <VerifyTable
        classes={classes}
        rebuilt={a.total}
        runtime={graphLogits}
        title={t('ex.classic.verifyTitle')}
        formula={t('ex.sm.rebuilt')}
        note={t('ex.classic.verifyNote')}
      />

      <p className="note exc-honest">{t('ex.sm.honest')}</p>
      <Narration text={narr} />
    </div>
  )
}
