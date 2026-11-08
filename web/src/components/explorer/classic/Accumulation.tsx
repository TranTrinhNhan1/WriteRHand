import { useMemo, useState } from 'react'
import { useI18n } from '../../../i18n'
import { usePlayback, type Playback } from '../../../engine/react'
import { softmax } from '../../../engine/reconstruct'
import { Segmented } from '../../Segmented'
import { PlaybackBar } from '../PlaybackBar'
import { LineChart } from './LineChart'
import { classRow, extent, fmt, leaderChanges, pixelOrder, rowCol, rowLeader, rowOf, runningScores, type Additive, type OrderKind } from './model'

/** Duration of one full pass over all pixels at 1x, in ms. */
const PASS_MS = 5000

export interface AccumulationState {
  orderKind: OrderKind
  setOrderKind: (o: OrderKind) => void
  order: Int32Array
  /** rank[j] = position of pixel j in the order (pixel j is applied at step rank[j] + 1) */
  rank: Int32Array
  running: Float64Array
  pb: Playback
  leader: number
  /** pixel applied by the current step (null at step 0) */
  current: number | null
}

/** Order, running sums and playback for the additive accumulation of one execution. */
export function useAccumulation(a: Additive, k: number, execKey: string): AccumulationState {
  const [orderKind, setOrderKind] = useState<OrderKind>('magnitude')
  const order = useMemo(() => pixelOrder(orderKind, classRow(a, k)), [a, k, orderKind])
  const rank = useMemo(() => {
    const r = new Int32Array(order.length)
    order.forEach((j, i) => (r[j] = i))
    return r
  }, [order])
  const running = useMemo(() => runningScores(a, order), [a, order])
  const pb = usePlayback(a.N, `${execKey}|${orderKind}|${orderKind === 'magnitude' ? k : ''}`, PASS_MS / a.N)
  const leader = rowLeader(running, a.K, pb.step)
  return { orderKind, setOrderKind, order, rank, running, pb, leader, current: pb.step > 0 ? order[pb.step - 1] : null }
}

interface Props {
  acc: AccumulationState
  a: Additive
  classes: string[]
  k: number
  width: number
  /** probabilities returned by the deployed graph (shown once every pixel is applied) */
  finalProbs: ArrayLike<number>
  /** "log joint" / "logit" */
  scoreName: string
  startName: string
  className?: string
}

/**
 * Running class scores while pixels are applied in the chosen order, then softmax. Shared by the
 * Naive Bayes view (log joints) and the softmax-regression view (logits).
 */
export function Accumulation({ acc, a, classes, k, width, finalProbs, scoreName, startName }: Props) {
  const { t } = useI18n()
  const { pb, running, leader, current, orderKind, setOrderKind } = acc
  const { K, N } = a
  const step = pb.step
  const domain = useMemo(() => extent(running), [running])
  const marks = useMemo(() => leaderChanges(running, K).filter((c) => c.step > 0), [running, K])
  const scores = rowOf(running, K, step)
  const done = step >= N
  const probs = done ? Array.from(finalProbs) : Array.from(softmax(scores))
  const yf = (v: number) => fmt(v, Math.abs(domain[1] - domain[0]) >= 50 ? 0 : 1)
  const where = current != null ? rowCol(current, width) : null
  const caption =
    step === 0
      ? t('ex.classic.capStart', { start: startName })
      : t('ex.classic.capStep', { s: step, n: N, r: where!.r, c: where!.c, leader: classes[leader] })
  const lastChange = marks.filter((m) => m.step <= step).pop()

  return (
    <section className="exc-acc" aria-label={t('ex.classic.accTitle')}>
      <div className="exc-acc__head">
        <h4 className="exc-h">{t('ex.classic.accTitle')}</h4>
        <Segmented
          legend={t('ex.classic.order')}
          value={orderKind}
          onChange={setOrderKind}
          options={[
            { value: 'magnitude', label: t('ex.classic.orderMag', { name: classes[k] }) },
            { value: 'raster', label: t('ex.classic.orderRaster') },
          ]}
        />
      </div>
      <p className="note">{orderKind === 'magnitude' ? t('ex.classic.orderMagNote', { name: classes[k], n: N }) : t('ex.classic.orderRasterNote', { n: N })}</p>
      <PlaybackBar pb={pb} caption={caption} unit={t('ex.classic.unitPixels')} />
      <div className="exc-acc__body">
        <div className="exc-acc__chart">
          <LineChart
            rows={running}
            K={K}
            steps={N}
            cursor={step}
            names={classes}
            highlight={k}
            leader={leader}
            future="hide"
            marks={marks.map((m) => m.step)}
            yFormat={yf}
            domain={domain}
            height={240}
            label={t('ex.classic.chartAria', { score: scoreName })}
            valueText={caption}
            onSeek={pb.seek}
          />
          <p className="exc-fig__cap">
            {t('ex.classic.chartCap', { score: scoreName, start: startName })}{' '}
            {lastChange ? t('ex.classic.lastChange', { s: lastChange.step, name: classes[lastChange.klass] }) : t('ex.classic.noChange')}
          </p>
        </div>
        <div className="exc-acc__table">
          <table className="exc-scores">
            <caption>{done ? t('ex.classic.probsFinal') : t('ex.classic.probsPartial', { s: step, n: N })}</caption>
            <thead>
              <tr>
                <th scope="col">{t('ex.classic.class')}</th>
                <th scope="col">{scoreName}</th>
                <th scope="col">{t('ex.classic.softmax')}</th>
              </tr>
            </thead>
            <tbody>
              {classes.map((name, c) => (
                <tr key={c} className={c === leader ? 'is-lead' : c === k ? 'is-sel' : undefined}>
                  <th scope="row">
                    <span className="exc-scores__name">{name}</span>
                    {c === leader && <span className="exc-tag">{t('ex.classic.leader')}</span>}
                  </th>
                  <td className="num">{fmt(scores[c], 2)}</td>
                  <td>
                    <span className="exc-bar" aria-hidden="true">
                      <span className="exc-bar__fill" style={{ transform: `scaleX(${probs[c]})` }} />
                    </span>
                    <span className="num exc-bar__v">{(probs[c] * 100).toFixed(probs[c] >= 0.9995 || probs[c] < 0.0005 ? 0 : 1)}%</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
