import { Pause, Play } from '@phosphor-icons/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useI18n } from '../../i18n'
import type { AdaBoostTrace } from '../../lib/trace'
import { HeatMap } from './HeatMap'

const SECONDS = 7 // a full play-through

/**
 * AdaBoost is a long committee of small decision trees. Each tree asks a handful of yes/no questions about single
 * pixels, lands in a leaf and casts a weighted vote for one digit. The slider adds trees one at a time.
 */
export function AdaView({ trace, pixels }: { trace: AdaBoostTrace; pixels: Float32Array }) {
  const { t } = useI18n()
  const total = trace.rounds.length
  const [round, setRound] = useState(1)
  const [playing, setPlaying] = useState(false)
  const raf = useRef(0)
  const roundRef = useRef(round)
  roundRef.current = round

  // running sums of the weighted votes after every number of trees
  const cum = useMemo(() => {
    const k = trace.classes.length
    const out = new Float64Array((total + 1) * k)
    for (let i = 0; i < total; i++) {
      const r = trace.rounds[i]
      for (let c = 0; c < k; c++) out[(i + 1) * k + c] = out[i * k + c]
      out[(i + 1) * k + trace.classes.indexOf(r.digit)] += r.alpha
    }
    return out
  }, [trace, total])

  useEffect(() => {
    if (!playing) return
    let last = performance.now()
    let acc = roundRef.current >= total ? 1 : roundRef.current
    const step = (now: number) => {
      acc += ((now - last) / 1000) * (total / SECONDS)
      last = now
      const next = Math.min(total, Math.floor(acc))
      setRound(next)
      if (next >= total) setPlaying(false)
      else raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, total])

  const k = trace.classes.length
  const cur = trace.rounds[round - 1]
  const sums = Array.from({ length: k }, (_, c) => cum[round * k + c])
  const sumAll = sums.reduce((a, b) => a + b, 0) || 1
  let lead = 0
  for (let c = 1; c < k; c++) if (sums[c] > sums[lead]) lead = c

  const seen = useMemo(() => {
    const counts = new Float32Array(784)
    for (let i = 0; i < round; i++) for (const s of trace.rounds[i].path) counts[s.feature] += 1
    return counts
  }, [trace, round])

  // a tree may test the same pixel at several depths; its marker then carries all of those step numbers
  const marks = useMemo(() => {
    const byPixel = new Map<number, number[]>()
    cur.path.forEach((s, i) => byPixel.set(s.feature, [...(byPixel.get(s.feature) ?? []), i + 1]))
    return [...byPixel].map(([index, steps]) => ({ index, label: steps.join(',') }))
  }, [cur])
  const fmt = (v: number) => v.toFixed(2)

  return (
    <div className="ada">
      <div className="ada__controls">
        <button
          type="button"
          className="btn btn--primary"
          onClick={() => {
            if (!playing && round >= total) setRound(1)
            setPlaying((p) => !p)
          }}
        >
          {playing ? <Pause size={18} weight="fill" aria-hidden="true" /> : <Play size={18} weight="fill" aria-hidden="true" />}
          {playing ? t('inside.ada.pause') : t('inside.ada.play')}
        </button>
        <label className="ada__range">
          <span className="ada__rangelabel num">{t('inside.ada.round', { t: round, total })}</span>
          <input
            type="range"
            min={1}
            max={total}
            value={round}
            onChange={(e) => {
              setPlaying(false)
              setRound(Number(e.target.value))
            }}
            aria-label={t('inside.ada.slider')}
          />
        </label>
      </div>

      <div className="ada__grid">
        <section className="ada__tree" aria-label={t('inside.ada.treeTitle', { t: round })}>
          <h4 className="flow__title">{t('inside.ada.treeTitle', { t: round })}</h4>
          <div className="ada__treebody">
            <div className="ada__pic">
              <HeatMap values={pixels} width={28} height={28} mode="gray" max={1} marks={marks} label={t('inside.ada.picAria')} />
              <p className="caption">{t('inside.ada.picCaption')}</p>
            </div>
            <ol className="ada__path">
              {cur.path.map((s, i) => (
                <li key={`${round}-${i}`} className="ada__node" style={{ '--depth': i } as React.CSSProperties}>
                  <span className="ada__no num">{i + 1}</span>
                  <div>
                    <p className="ada__q">{t('inside.ada.pixel', { r: Math.floor(s.feature / 28), c: s.feature % 28 })}</p>
                    <p className="ada__a num">
                      {s.goLeft ? t('inside.ada.less', { v: fmt(s.value), th: fmt(s.threshold) }) : t('inside.ada.more', { v: fmt(s.value), th: fmt(s.threshold) })}
                    </p>
                    <p className="ada__skip">{t('inside.ada.other', { n: s.otherSize })}</p>
                  </div>
                </li>
              ))}
              <li className="ada__leaf">
                <span className="ada__no ada__no--leaf num" aria-hidden="true">
                  {cur.digit}
                </span>
                <div>
                  <p className="ada__q">{t('inside.ada.leaf', { digit: cur.digit })}</p>
                  <p className="ada__a num">{t('inside.ada.weight', { a: cur.alpha.toFixed(2) })}</p>
                </div>
              </li>
            </ol>
          </div>
        </section>

        <section className="ada__votes" aria-label={t('inside.ada.votesTitle')}>
          <h4 className="flow__title">{t('inside.ada.votesTitle')}</h4>
          <ul className="vbars">
            {trace.classes.map((d, c) => (
              <li key={d} className={c === lead ? 'vbars__row is-lead' : 'vbars__row'}>
                <span className="vbars__digit num">{d}</span>
                <span className="vbars__track" aria-hidden="true">
                  <span className="vbars__fill" style={{ transform: `scaleX(${sums[c] / sumAll})` }} />
                </span>
                <span className="vbars__value num">{((sums[c] / sumAll) * 100).toFixed(0)}%</span>
              </li>
            ))}
          </ul>
          <p className="ada__lead" aria-live="off">
            {t('inside.ada.lead', { t: round, digit: trace.classes[lead] })}
          </p>
          <div className="ada__attn">
            <HeatMap values={seen} width={28} height={28} mode="active" label={t('inside.ada.attnAria')} className="heat--small" />
            <p className="caption">{t('inside.ada.attn', { t: round })}</p>
          </div>
        </section>
      </div>
    </div>
  )
}
