import { useMemo } from 'react'
import { useI18n } from '../../../i18n'
import { denseNeuron } from '../../../engine/reconstruct'
import { HeatMap } from '../HeatMap'
import { fx, int } from './format'
import { strongest } from './model'
import { CheckRow, Hint, Term } from './ui'

export type SpatialInput = { kind: 'image'; side: number } | { kind: 'maps'; c: number; h: number; w: number } | null

interface Props {
  /** values that feed the layer (the graph's output of the previous layer) */
  input: Float32Array
  /** symbol of the input values: "x" (standardised pixels) or "a" (activations) */
  sym: string
  W: Float32Array
  b: Float32Array
  nIn: number
  nOut: number
  j: number
  /** values the graph returned for this neuron: pre-activation and, after a ReLU, the activation */
  zGraph: number
  aGraph: number | null
  spatial: SpatialInput
  /** the source of a selected connection, and how to name a source index */
  picked: number | null
  onPick: (i: number | null) => void
  sourceName: (i: number) => string
}

const LIST = 10

/**
 * One output neuron of a dense layer, rebuilt from its parts: every x_i·W_ij, the positive and
 * negative sums, the bias, z next to the value the graph returned, then the ReLU. These sums are
 * an exact decomposition of z for this input; they do not say why the model decided.
 */
export function DenseInspector({ input, sym, W, b, nIn, nOut, j, zGraph, aGraph, spatial, picked, onPick, sourceName }: Props) {
  const { t } = useI18n()
  const n = useMemo(() => denseNeuron(input, W, b, nIn, nOut, j), [input, W, b, nIn, nOut, j])
  const weights = useMemo(() => {
    const w = new Float32Array(nIn)
    for (let i = 0; i < nIn; i++) w[i] = W[i * nOut + j]
    return w
  }, [W, nIn, nOut, j])
  const all = useMemo(() => Array.from({ length: nIn }, (_, i) => i), [nIn])
  const top = useMemo(() => strongest(all, (i) => n.terms[i], LIST), [all, n])
  const nPos = n.terms.reduce((c, v) => c + (v > 0 ? 1 : 0), 0)
  const nNeg = n.terms.reduce((c, v) => c + (v < 0 ? 1 : 0), 0)
  const listed = top.reduce((s, i) => s + n.terms[i], 0)
  const maxAbs = Math.max(...top.map((i) => Math.abs(n.terms[i])), 1e-12)
  const perChannel = useMemo(() => {
    if (spatial?.kind !== 'maps') return null
    const hw = spatial.h * spatial.w
    return Array.from({ length: spatial.c }, (_, c) => {
      let s = 0
      for (let k = 0; k < hw; k++) s += n.terms[c * hw + k]
      return s
    })
  }, [spatial, n])

  return (
    <div className="ex-dense">
      <dl className="ex-terms">
        <Term label={t('ex.mlp.sumPos', { n: int(nPos), sym })} value={n.positive} />
        <Term label={t('ex.mlp.sumNeg', { n: int(nNeg), sym })} value={n.negative} />
        <Term label={t('ex.mlp.bias', { j })} value={n.bias} />
      </dl>
      <CheckRow label={<span className="num">z = Σ {sym}·W + b</span>} rebuilt={n.z} runtime={zGraph} />
      {aGraph != null && <CheckRow label={<span className="num">a = max(0, z)</span>} rebuilt={Math.max(0, n.z)} runtime={aGraph} />}

      {spatial?.kind === 'image' && (
        <div className="ex-dense__maps">
          {[
            { v: input, l: t('ex.mlp.mapInput', { sym }) },
            { v: weights, l: t('ex.mlp.mapWeights', { j }) },
            { v: n.terms, l: t('ex.mlp.mapTerms', { sym }) },
          ].map((m) => (
            <figure key={m.l} className="ex-dense__map">
              <HeatMap values={m.v} width={spatial.side} height={spatial.side} mode="diverging" label={m.l} selected={picked} onSelect={onPick} />
              <figcaption>{m.l}</figcaption>
            </figure>
          ))}
        </div>
      )}
      {spatial?.kind === 'maps' && perChannel && (
        <div className="ex-dense__chan">
          <p className="ex-dense__cap">{t('ex.cnn.perChannel', { c: spatial.c, h: spatial.h, w: spatial.w })}</p>
          <div className="ex-thumbs" role="list">
            {perChannel.map((s, c) => {
              const hw = spatial.h * spatial.w
              const inside = picked != null && picked >= c * hw && picked < (c + 1) * hw ? picked - c * hw : null
              return (
                <figure key={c} className="ex-thumb" role="listitem">
                  <HeatMap
                    values={n.terms.subarray(c * hw, (c + 1) * hw)}
                    width={spatial.w}
                    height={spatial.h}
                    mode="diverging"
                    max={maxAbs}
                    label={t('ex.cnn.chanTerms', { c, s: fx(s, 3, true) })}
                    selected={inside}
                    onSelect={(k) => onPick(c * hw + k)}
                  />
                  <figcaption className="num">
                    c{c} <b data-sign={s >= 0 ? 'pos' : 'neg'}>{fx(s, 2, true)}</b>
                  </figcaption>
                </figure>
              )
            })}
          </div>
        </div>
      )}

      <div className="ex-contrib">
        <p className="ex-dense__cap">{t('ex.mlp.topTerms', { k: top.length, n: int(nIn), sym, s: fx(listed, 3, true) })}</p>
        <ol className="ex-contrib__list">
          {top.map((i) => {
            const v = n.terms[i]
            return (
              <li key={i}>
                <button type="button" className="ex-contrib__row" aria-pressed={picked === i} onClick={() => onPick(picked === i ? null : i)}>
                  <span className="ex-contrib__name">{sourceName(i)}</span>
                  <span className="num ex-contrib__calc">
                    {fx(input[i], 3)} × {fx(weights[i], 3)}
                  </span>
                  <span className="ex-contrib__bar" data-sign={v >= 0 ? 'pos' : 'neg'} aria-hidden="true">
                    <span style={{ transform: `scaleX(${Math.abs(v) / maxAbs})` }} />
                  </span>
                  <span className="num ex-contrib__v">{fx(v, 4, true)}</span>
                </button>
              </li>
            )
          })}
        </ol>
      </div>

      {picked != null && picked < nIn && (
        <div className="ex-conn" aria-live="polite">
          <p className="ex-dense__cap">{t('ex.mlp.conn', { src: sourceName(picked), j })}</p>
          <dl className="ex-terms">
            <Term label={`${sym}`} value={input[picked]} />
            <Term label="W" value={weights[picked]} />
            <Term label={`${sym}·W`} value={n.terms[picked]} strong />
          </dl>
          <p className="ex-hint">{n.terms[picked] > 0 ? t('ex.mlp.connPos') : n.terms[picked] < 0 ? t('ex.mlp.connNeg') : t('ex.mlp.connZero')}</p>
        </div>
      )}
      <Hint>{t('ex.mlp.exact')}</Hint>
    </div>
  )
}
