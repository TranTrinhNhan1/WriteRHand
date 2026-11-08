import { useMemo, useState } from 'react'
import { useI18n } from '../../i18n'
import { logSumExp, nbClassTerms } from '../../engine/reconstruct'
import { HeatMap } from './HeatMap'
import type { ExplorerViewProps } from './types'
import { Accumulation, useAccumulation } from './classic/Accumulation'
import { GaussPlot } from './classic/GaussPlot'
import { absQuantile, classRow, extent, fmt, makeAdditive, rowCol, rowOf } from './classic/model'
import { ClassPicker, Legend, MapFigure, Narration, Readout, VerifyTable } from './classic/parts'
import { imageShape, layerOf, runtimeOf } from './classic/trace'

/**
 * Gaussian Naive Bayes, computed in the open: the class means and variances the model learned, the
 * per-pixel log-density terms for the current input, their accumulation into the log joint of every
 * class, softmax to the posterior, and a check against the values returned by the deployed graph.
 */
export default function NaiveBayesView({ classes, loaded, execution }: ExplorerViewProps) {
  const { t } = useI18n()
  const { trace } = loaded
  const layer = layerOf(trace, 'gaussian_loglik')
  const { h, w } = imageShape(trace)
  const params = loaded.params!
  const mean = params.get(layer.params!.mean)
  const variance = params.get(layer.params!.variance)
  const logPrior = params.get(layer.params!.log_prior)
  const K = classes.length
  const N = layer.n_features ?? mean.length / K
  const x = runtimeOf(trace, execution, 'standardized_input')
  const graphLogJoint = runtimeOf(trace, execution, 'log_joint')

  const [kState, setK] = useState(execution.top)
  const k = Math.min(kState, K - 1)
  const [picked, setPicked] = useState<number | null>(null)
  const [hover, setHover] = useState<number | null>(null)

  const a = useMemo(
    () =>
      makeAdditive(K, N, (c) => {
        const r = nbClassTerms(x, mean, variance, logPrior, c, N)
        return { terms: r.perPixel, start: r.logPrior }
      }),
    [K, N, x, mean, variance, logPrior],
  )
  const acc = useAccumulation(a, k, `${execution.seq}`)
  const step = acc.pb.step
  const row = classRow(a, k)

  // colour scales: fixed per model / class so nothing rescales while the animation runs
  const scales = useMemo(() => {
    const [mlo, mhi] = extent(mean)
    const [xlo, xhi] = extent(x)
    const [, vhi] = extent(variance)
    return { std: Math.max(Math.abs(mlo), Math.abs(mhi), Math.abs(xlo), Math.abs(xhi)), varMax: vhi }
  }, [mean, variance, x])
  const termMax = useMemo(() => absQuantile(row, 0.99), [row])
  const [vlo, vhi] = useMemo(() => extent(variance.subarray(k * N, (k + 1) * N)), [variance, k, N])
  const applied = useMemo(() => {
    const out = new Float64Array(N)
    for (let i = 0; i < step; i++) out[acc.order[i]] = row[acc.order[i]]
    return out
  }, [N, step, acc.order, row])

  const j = hover ?? picked ?? acc.order[0]
  const { r, c } = rowCol(j, w)
  const mu = mean[k * N + j]
  const v = variance[k * N + j]
  const logNorm = -0.5 * Math.log(2 * Math.PI * v)
  const quad = (-0.5 * (x[j] - mu) * (x[j] - mu)) / v
  const mus = useMemo(() => Float64Array.from({ length: K }, (_, q) => mean[q * N + j]), [K, N, j, mean])
  const vars = useMemo(() => Float64Array.from({ length: K }, (_, q) => variance[q * N + j]), [K, N, j, variance])
  const evidence = logSumExp(graphLogJoint)
  const rank = acc.rank[j] + 1

  const mapProps = {
    width: w,
    height: h,
    selected: j,
    onSelect: (p: number) => setPicked(p),
    onHover: setHover,
  }
  const scoreAt = rowOf(acc.running, K, step)
  const narr = acc.pb.playing
    ? ''
    : step === 0
      ? t('ex.nb.narrStart')
      : t('ex.nb.narr', { s: step, n: N, r: rowCol(acc.current!, w).r, c: rowCol(acc.current!, w).c, leader: classes[acc.leader], v: fmt(scoreAt[acc.leader], 1) })

  return (
    <div className="exc">
      <p className="exc-intro">{t('ex.nb.intro')}</p>

      <section className="exc-sec" aria-label={t('ex.nb.classTitle')}>
        <h4 className="exc-h">{t('ex.nb.classTitle')}</h4>
        <ClassPicker
          classes={classes}
          value={k}
          onChange={setK}
          width={w}
          height={h}
          label={t('ex.nb.classTitle')}
          thumb={(q) => ({ values: mean.subarray(q * N, (q + 1) * N), mode: 'diverging', max: scales.std })}
          sub={(q) => fmt(logPrior[q], 2)}
        />
      </section>

      <div className="exc-grid">
        <section className="exc-maps" aria-label={t('ex.nb.mapsTitle', { name: classes[k] })}>
          <h4 className="exc-h">{t('ex.nb.mapsTitle', { name: classes[k] })}</h4>
          <div className="exc-maps__row">
            <MapFigure title={t('ex.nb.mapX')} caption={t('ex.nb.mapXCap')}>
              <HeatMap {...mapProps} values={x} mode="diverging" max={scales.std} label={t('ex.nb.mapXAria')} />
            </MapFigure>
            <MapFigure title={t('ex.nb.mapMean', { name: classes[k] })} caption={t('ex.nb.mapMeanCap')}>
              <HeatMap {...mapProps} values={mean.subarray(k * N, (k + 1) * N)} mode="diverging" max={scales.std} label={t('ex.nb.mapMeanAria', { name: classes[k] })} />
            </MapFigure>
            <MapFigure title={t('ex.nb.mapVar', { name: classes[k] })} caption={t('ex.nb.mapVarCap', { lo: fmt(vlo, 3), hi: fmt(vhi, 3) })}>
              <HeatMap {...mapProps} values={variance.subarray(k * N, (k + 1) * N)} mode="active" max={scales.varMax} label={t('ex.nb.mapVarAria', { name: classes[k] })} />
            </MapFigure>
            <MapFigure title={t('ex.nb.mapTerm', { name: classes[k] })} caption={t('ex.nb.mapTermCap', { s: step, n: N, m: fmt(termMax, 2) })}>
              <HeatMap
                {...mapProps}
                values={applied}
                mode="diverging"
                max={termMax}
                box={acc.current != null && step < N ? { x0: acc.current % w, y0: Math.floor(acc.current / w), x1: (acc.current % w) + 1, y1: Math.floor(acc.current / w) + 1 } : null}
                label={t('ex.nb.mapTermAria', { name: classes[k] })}
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

        <section className="exc-inspect" aria-label={t('ex.nb.inspectTitle')} aria-live="off">
          <h4 className="exc-h">{t('ex.classic.pixelTitle', { r, c, name: classes[k] })}</h4>
          <Readout
            rows={[
              { sym: 'p', label: t('ex.classic.rawPixel'), value: fmt(execution.pixels[j], 3) },
              { sym: 'x', label: t('ex.classic.stdPixel'), value: fmt(x[j], 4) },
              { sym: 'μ', label: t('ex.nb.mu'), value: fmt(mu, 4) },
              { sym: 'σ²', label: t('ex.nb.var'), value: fmt(v, 4) },
              { sym: '−½ log 2πσ²', label: t('ex.nb.logNorm'), value: fmt(logNorm, 4) },
              { sym: '−(x−μ)²/2σ²', label: t('ex.nb.quad'), value: fmt(quad, 4) },
              { sym: 'log N', label: t('ex.nb.term'), value: fmt(logNorm + quad, 4), strong: true },
              { sym: 'log p(k)', label: t('ex.nb.prior'), value: fmt(logPrior[k], 4) },
            ]}
          />
          <p className="note">{rank <= step ? t('ex.classic.applied', { s: rank }) : t('ex.classic.notApplied', { s: rank })}</p>
          <GaussPlot mus={mus} vars={vars} x={x[j]} k={k} className={classes[k]} />
        </section>
      </div>

      <Accumulation acc={acc} a={a} classes={classes} k={k} width={w} finalProbs={execution.probs} scoreName={t('ex.nb.score')} startName={t('ex.nb.start')} />

      <section className="exc-sec exc-post" aria-label={t('ex.nb.postTitle')}>
        <h4 className="exc-h">{t('ex.nb.postTitle')}</h4>
        <p className="exc-formula num">{t('ex.nb.postFormula', { e: fmt(evidence, 3) })}</p>
        <p className="note">{t('ex.nb.postNote', { name: classes[execution.top], p: `${(execution.probs[execution.top] * 100).toFixed(2)}%` })}</p>
      </section>

      <VerifyTable
        classes={classes}
        rebuilt={a.total}
        runtime={graphLogJoint}
        title={t('ex.classic.verifyTitle')}
        formula={t('ex.nb.rebuilt')}
        note={t('ex.classic.verifyNote')}
      />

      <p className="note exc-honest">{t('ex.nb.honest')}</p>
      <Narration text={narr} />
    </div>
  )
}
