import { useMemo } from 'react'
import { useI18n } from '../../i18n'
import type { NaiveBayesTrace, SoftmaxTrace } from '../../lib/trace'
import { N_PIXELS } from '../../lib/trace'
import { HeatMap } from './HeatMap'

const K = 10

/**
 * Naive Bayes and softmax regression share one picture: for every digit class there is a pixel template (what that
 * class looks like to the model) and an evidence map (how each pixel of YOUR drawing speaks for or against that class).
 * Adding up the evidence map, plus a constant, gives the class score.
 */
export function ClassMaps({ trace, probs, pixels }: { trace: NaiveBayesTrace | SoftmaxTrace; probs: ArrayLike<number>; pixels: Float32Array }) {
  const { t } = useI18n()
  const isNb = trace.kind === 'naive_bayes'

  const { templates, evidence, scores, tMax, eMax, win } = useMemo(() => {
    const templates = new Float32Array(K * N_PIXELS)
    const evidence = new Float32Array(K * N_PIXELS)
    if (trace.kind === 'naive_bayes') {
      const { x, mean, variance } = trace
      // Per-pixel log-likelihood of the drawing under each class, relative to the average class: only differences
      // between classes decide the answer, and they sum to the difference between the class scores.
      const ll = new Float32Array(K * N_PIXELS)
      for (let c = 0; c < K; c++)
        for (let j = 0; j < N_PIXELS; j++) {
          const v = variance[c * N_PIXELS + j]
          const d = x[j] - mean[c * N_PIXELS + j]
          ll[c * N_PIXELS + j] = -0.5 * (Math.log(2 * Math.PI * v) + (d * d) / v)
          templates[c * N_PIXELS + j] = mean[c * N_PIXELS + j]
        }
      for (let j = 0; j < N_PIXELS; j++) {
        let avg = 0
        for (let c = 0; c < K; c++) avg += ll[c * N_PIXELS + j]
        avg /= K
        for (let c = 0; c < K; c++) evidence[c * N_PIXELS + j] = ll[c * N_PIXELS + j] - avg
      }
    } else {
      const { x, weights } = trace
      for (let c = 0; c < K; c++)
        for (let j = 0; j < N_PIXELS; j++) {
          templates[c * N_PIXELS + j] = weights[j * K + c]
          evidence[c * N_PIXELS + j] = x[j] * weights[j * K + c]
        }
    }
    let tMax = 0
    let eMax = 0
    for (let i = 0; i < templates.length; i++) tMax = Math.max(tMax, Math.abs(templates[i]))
    // Clip the evidence scale to a high percentile-like level so a few extreme pixels do not wash everything out.
    const abs = Float32Array.from(evidence, Math.abs).sort()
    eMax = abs[Math.floor(abs.length * 0.995)] || 1
    const logits = trace.logits
    let best = 0
    for (let c = 1; c < K; c++) if (logits[c] > logits[best]) best = c
    const scores = Array.from(logits, (v) => v - logits[best])
    return { templates, evidence, scores, tMax, eMax, win: best }
  }, [trace])

  const nbTemplateMin = useMemo(() => {
    if (!isNb) return 0
    let lo = Infinity
    for (let i = 0; i < templates.length; i++) lo = Math.min(lo, templates[i])
    return lo
  }, [isNb, templates])

  return (
    <div className="xray">
      <ol className="xray__steps">
        <li>
          <b className="num">1</b>
          <span>{t(isNb ? 'inside.nb.step1' : 'inside.sm.step1')}</span>
        </li>
        <li>
          <b className="num">2</b>
          <span>{t(isNb ? 'inside.nb.step2' : 'inside.sm.step2')}</span>
        </li>
        <li>
          <b className="num">3</b>
          <span>{t(isNb ? 'inside.nb.step3' : 'inside.sm.step3')}</span>
        </li>
      </ol>

      <div className="xray__row">
        <figure className="xray__input">
          <HeatMap values={pixels} width={28} height={28} mode="gray" max={1} label={t('inside.inputAria')} />
          <figcaption>{t('inside.input')}</figcaption>
        </figure>
        <ul className="xray__classes">
          {Array.from({ length: K }, (_, c) => (
            <li key={c} className={`xcls${c === win ? ' is-win' : ''}`}>
              <div className="xcls__head">
                <span className="xcls__digit num">{c}</span>
                <span className="xcls__score num">{c === win ? t('inside.best') : scores[c].toFixed(0)}</span>
              </div>
              <HeatMap
                values={isNb ? Float32Array.from(templates.subarray(c * N_PIXELS, (c + 1) * N_PIXELS), (v) => v - nbTemplateMin) : templates.subarray(c * N_PIXELS, (c + 1) * N_PIXELS)}
                width={28}
                height={28}
                mode={isNb ? 'gray' : 'diverging'}
                max={isNb ? undefined : tMax}
                label={t(isNb ? 'inside.nb.templateAria' : 'inside.sm.templateAria', { digit: c })}
              />
              <HeatMap
                values={evidence.subarray(c * N_PIXELS, (c + 1) * N_PIXELS)}
                width={28}
                height={28}
                mode="diverging"
                max={eMax}
                label={t('inside.evidenceAria', { digit: c })}
              />
              <span className="xcls__prob num" aria-label={t('rec.bars.row', { digit: c, value: `${(probs[c] * 100).toFixed(1)}%` })}>
                {(probs[c] * 100).toFixed(probs[c] >= 0.995 || probs[c] < 0.0005 ? 0 : 1)}%
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div className="xray__legend" aria-hidden="true">
        <span>
          <i className="swatch swatch--pos" />
          {t('inside.legend.for')}
        </span>
        <span>
          <i className="swatch swatch--neg" />
          {t('inside.legend.against')}
        </span>
        <span className="xray__rows">
          {t('inside.legend.rows', { a: t(isNb ? 'inside.nb.rowA' : 'inside.sm.rowA'), b: t('inside.rowB') })}
        </span>
      </div>
    </div>
  )
}
