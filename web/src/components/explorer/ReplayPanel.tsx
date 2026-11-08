import { ArrowClockwise, CheckCircle, ClockCounterClockwise, WarningCircle, XCircle } from '@phosphor-icons/react'
import { useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { useI18n, type Key } from '../../i18n'
import type { ModelKind, ReplayData, ReplayTensor } from '../../engine/contract'
import { useBundle, usePlayback, useReplay } from '../../engine/react'
import { bytes } from '../../lib/format'
import { HeatMap } from './HeatMap'
import { PlaybackBar } from './PlaybackBar'
import { fx, int, shape } from './neural/format'
import { column, correctCount, predicted, samplePixels, seriesAt, snapshotAt, tensorLayout, tensorMax } from './neural/replay'

interface Props {
  dataset: string
  model: ModelKind
}

/**
 * Training replay: the genuine record of the run that produced the model (measured series,
 * checkpointed tensors, recorded predictions on fixed training images). Nothing is simulated;
 * when a model has no recording the panel says so.
 */
export default function ReplayPanel({ dataset, model }: Props) {
  const [attempt, setAttempt] = useState(0)
  return <ReplayBody key={`${dataset}/${model}/${attempt}`} dataset={dataset} model={model} onRetry={() => setAttempt((a) => a + 1)} />
}

function ReplayBody({ dataset, model, onRetry }: Props & { onRetry: () => void }) {
  const { t } = useI18n()
  const { manifest } = useBundle()
  const ds = manifest.datasets[dataset]
  const entry = ds.models[model]!
  const rp = useReplay(dataset, model, true)
  const badge = (
    <p className="ex-badge ex-badge--rec">
      <ClockCounterClockwise size={14} aria-hidden="true" />
      <b>{t('ex.replay.badge')}</b>
      <span>{t('ex.replay.badgeSub')}</span>
    </p>
  )
  if (rp.state === 'loading' || rp.state === 'idle') {
    return (
      <div className="ex-replay">
        {badge}
        <div className="ex-state ex-state--pending" aria-busy="true">
          <p className="caption" role="status">
            {t('ex.replay.loading', { size: entry.replay ? bytes(entry.replay.size_bytes) : '' })}
          </p>
          <div className="skeleton skeleton--line" />
          <div className="skeleton skeleton--block" />
        </div>
      </div>
    )
  }
  if (rp.state === 'error') {
    return (
      <div className="ex-replay">
        {badge}
        <div className="ex-state ex-state--error" role="alert">
          <p className="msg msg--error">{t(rp.error!.key, rp.error!.params)}</p>
          <button type="button" className="btn btn--quiet" onClick={onRetry}>
            <ArrowClockwise size={16} aria-hidden="true" />
            {t('common.retry')}
          </button>
        </div>
      </div>
    )
  }
  if (rp.state === 'none') {
    return (
      <div className="ex-replay">
        {badge}
        <div className="ex-state ex-state--empty">
          <ClockCounterClockwise size={28} aria-hidden="true" />
          <p className="label">{t('ex.replay.noneTitle')}</p>
          <p className="caption">{model === 'naive_bayes' ? t('ex.replay.noneNb') : t('ex.replay.noneMissing', { model: entry.title })}</p>
          <p className="caption">{t('ex.replay.noneHonest')}</p>
        </div>
      </div>
    )
  }
  return (
    <div className="ex-replay">
      {badge}
      <ReplayView r={rp.replay!} classes={ds.classes} vote={entry.trace.output.kind === 'vote_share'} />
    </div>
  )
}

// --------------------------------------------------------------------------- the recorded run
const SERIES_KEYS = ['loss', 'val_loss', 'inner_val_loss', 'val_acc', 'train_acc', 'train_err_weighted', 'alpha'] as const

function ReplayView({ r, classes, vote }: { r: ReplayData; classes: string[]; vote: boolean }) {
  const { t } = useI18n()
  const rounds = r.model === 'adaboost'
  const unit = rounds ? t('ex.replay.round') : t('ex.replay.epoch')
  const last = r.snapshots.length - 1
  const pb = usePlayback(last, `${r.dataset}/${r.model}`, 900, false)
  const k = Math.min(pb.step, last)
  const snap = r.snapshots[k]
  const at = seriesAt(r, snap.epoch)
  const deployed = typeof r.deployed_epoch === 'number' ? r.deployed_epoch : null
  const deployedIdx = snapshotAt(r, deployed)
  const correct = correctCount(snap, r.samples.labels)
  const seriesName = (key: string) => ((SERIES_KEYS as readonly string[]).includes(key) ? t(`ex.replay.series.${key}` as Key) : key)
  const marks = [
    r.best_epoch != null ? { x: r.best_epoch, label: t('ex.replay.best') } : null,
    r.stopped_epoch != null && r.stopped_epoch !== r.best_epoch ? { x: r.stopped_epoch, label: t('ex.replay.stopped') } : null,
  ].filter(Boolean) as { x: number; label: string }[]
  const side = Math.round(Math.sqrt(r.samples.pixels[0]?.length ?? 784))
  const caption = t('ex.replay.caption', { unit, e: snap.epoch, k: k + 1, n: r.snapshots.length })

  return (
    <div className="ex-view ex-replay__view">
      <dl className="ex-replay__facts">
        <div>
          <dt>{t('ex.replay.seed')}</dt>
          <dd className="num">{r.seed}</dd>
        </div>
        <div>
          <dt>{t('ex.replay.recorded', { unit })}</dt>
          <dd className="num">{t('ex.replay.recordedVal', { n: int(r.epochs.length), s: r.snapshots.length })}</dd>
        </div>
        <div>
          <dt>{t('ex.replay.best')}</dt>
          <dd className="num">{r.best_epoch ?? t('ex.replay.noBest')}</dd>
        </div>
        <div>
          <dt>{t('ex.replay.stopped')}</dt>
          <dd className="num">{r.stopped_epoch ?? '–'}</dd>
        </div>
        <div>
          <dt>{t('ex.replay.deployed')}</dt>
          <dd className="num">{deployed ?? t('ex.replay.deployedUnknown')}</dd>
        </div>
      </dl>
      <p className="ex-replay__match" data-ok={r.final_matches_deployed}>
        {r.final_matches_deployed ? <CheckCircle size={16} weight="fill" aria-hidden="true" /> : <WarningCircle size={16} weight="fill" aria-hidden="true" />}
        {deployed != null
          ? t(r.final_matches_deployed ? 'ex.replay.matchYes' : 'ex.replay.matchNo', { unit, e: deployed })
          : t(r.final_matches_deployed ? 'ex.replay.matchYesFinal' : 'ex.replay.matchNoFinal')}
      </p>
      <p className="ex-hint">
        {t('ex.replay.noteLabel')} “{r.note}”
      </p>

      <div className="ex-view__bar">
        <PlaybackBar pb={pb} caption={caption} unit={t('ex.replay.snapshots')} />
        <p className="visually-hidden" aria-live="polite">
          {pb.playing ? '' : caption}
        </p>
      </div>

      <section className="ex-panel" aria-labelledby="ex-rp-series">
        <header className="ex-panel__head">
          <h3 className="ex-panel__title" id="ex-rp-series">
            {t('ex.replay.seriesTitle')}
          </h3>
          <p className="ex-panel__sub">{t('ex.replay.seriesSub', { unit })}</p>
        </header>
        <div className="ex-series">
          {Object.entries(r.series).map(([key, v]) => (
            <SeriesChart key={key} name={seriesName(key)} xs={r.epochs} ys={v} cursor={snap.epoch} marks={marks} deployed={deployed} unit={unit} />
          ))}
        </div>
        <div className="ex-snapmarks" role="group" aria-label={t('ex.replay.jump')}>
          {r.snapshots.map((s, i) => (
            <button key={s.epoch} type="button" className="ex-snapmark" aria-pressed={i === k} data-deployed={i === deployedIdx} onClick={() => pb.seek(i)}>
              <span className="num">{s.epoch}</span>
              {i === deployedIdx && <span className="ex-snapmark__tag">{t('ex.replay.deployedTag')}</span>}
            </button>
          ))}
        </div>
      </section>

      <section className="ex-panel" aria-labelledby="ex-rp-snap">
        <header className="ex-panel__head">
          <h3 className="ex-panel__title" id="ex-rp-snap">
            {snap.epoch === 0 ? t(rounds ? 'ex.replay.snapInitRounds' : 'ex.replay.snapInit') : t('ex.replay.snapTitle', { unit, e: snap.epoch })}
            {k === deployedIdx && <span className="ex-tag">{t('ex.replay.deployedTag')}</span>}
          </h3>
          <p className="ex-panel__sub">{rounds ? t('ex.replay.snapSubRounds', { e: snap.epoch }) : t('ex.replay.snapSub')}</p>
        </header>
        {at ? (
          <dl className="ex-replay__vals">
            {Object.entries(at).map(([key, v]) => (
              <div key={key}>
                <dt>{seriesName(key)}</dt>
                <dd className="num">{fx(v, 4)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="ex-hint">{t('ex.replay.noValues', { unit })}</p>
        )}

        {Object.keys(snap.tensors).length > 0 ? (
          Object.entries(snap.tensors).map(([name, tensor]) => <TensorView key={name} name={name} tensor={tensor} max={tensorMax(r.snapshots, name)} side={side} />)
        ) : (
          <p className="ex-hint">{t('ex.replay.noTensors')}</p>
        )}

        <div className="ex-block">
          <h4 className="ex-block__title">{t('ex.replay.samplesTitle', { n: r.samples.labels.length })}</h4>
          <p className="ex-hint">
            {t(vote ? 'ex.replay.samplesVote' : 'ex.replay.samplesProb', { c: correct, n: r.samples.labels.length })}
          </p>
          <ol className="ex-samples">
            {r.samples.labels.map((label, i) => {
              const row = snap.proba[i]
              const p = predicted(row)
              const ok = p === label
              return (
                <li key={i} className="ex-sample" data-ok={p < 0 ? 'none' : ok}>
                  <HeatMap values={samplePixels(r.samples.pixels[i])} width={side} height={side} mode="gray" max={1} label={t('ex.replay.sampleAria', { i: r.samples.indices[i], label: classes[label] ?? label })} />
                  <div className="ex-sample__text">
                    <span className="ex-sample__label">
                      {t('ex.replay.label')} <b>{classes[label] ?? label}</b>
                    </span>
                    <span className="ex-sample__pred">
                      {p < 0 ? null : ok ? <CheckCircle size={14} weight="fill" aria-hidden="true" /> : <XCircle size={14} weight="fill" aria-hidden="true" />}
                      {p < 0 ? t('ex.replay.noVote') : t(ok ? 'ex.replay.predOk' : 'ex.replay.predBad', { c: classes[p] ?? p })}
                    </span>
                  </div>
                  <span className="ex-sample__bars" aria-hidden="true">
                    {row.map((v, j) => (
                      <span key={j} data-top={j === p} data-true={j === label} style={{ '--h': `${Math.max(0, Math.min(1, v)) * 100}%` } as CSSProperties} title={`${classes[j] ?? j}: ${(v * 100).toFixed(1)}%`} />
                    ))}
                  </span>
                  <span className="visually-hidden">{row.map((v, j) => `${classes[j] ?? j} ${(v * 100).toFixed(1)}%`).join(', ')}</span>
                </li>
              )
            })}
          </ol>
        </div>
      </section>
    </div>
  )
}

// --------------------------------------------------------------------------- small multiples of the measured series
function SeriesChart({ name, xs, ys, cursor, marks, deployed, unit }: { name: string; xs: number[]; ys: number[]; cursor: number; marks: { x: number; label: string }[]; deployed: number | null; unit: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(0)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    setW(el.getBoundingClientRect().width)
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const H = 120
  const M = { l: 44, r: 8, t: 8, b: 20 }
  const finite = ys.filter(Number.isFinite)
  const lo = Math.min(...finite)
  const hi = Math.max(...finite)
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08
  const y0 = lo - pad
  const y1 = hi + pad
  const x0 = Math.min(0, xs[0] ?? 0)
  const x1 = xs[xs.length - 1] ?? 1
  const sx = (x: number) => M.l + ((x - x0) / Math.max(1e-9, x1 - x0)) * Math.max(10, w - M.l - M.r)
  const sy = (y: number) => M.t + (1 - (y - y0) / Math.max(1e-12, y1 - y0)) * (H - M.t - M.b)
  const path = useMemo(() => xs.map((x, i) => (Number.isFinite(ys[i]) ? `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}` : null)).filter(Boolean).join(' '), [xs, ys, w]) // eslint-disable-line react-hooks/exhaustive-deps
  const ci = xs.indexOf(cursor)
  const label = `${name}: ${xs.length} ${unit}, ${fx(ys[0], 4)} … ${fx(ys[ys.length - 1], 4)}`
  return (
    <figure className="ex-chart" ref={ref}>
      <figcaption className="ex-chart__name">
        {name} {ci >= 0 && <span className="num">= {fx(ys[ci], 4)}</span>}
      </figcaption>
      {w > 0 && (
        <svg width={w} height={H} role="img" aria-label={label}>
          {[y0 + pad, y1 - pad].map((v, i) => (
            <g key={i}>
              <line x1={M.l} x2={w - M.r} y1={sy(v)} y2={sy(v)} className="ex-chart__grid" />
              <text x={M.l - 6} y={sy(v) + 3} textAnchor="end" className="ex-chart__tick">
                {fx(v, Math.abs(v) >= 10 ? 1 : 3)}
              </text>
            </g>
          ))}
          <text x={M.l} y={H - 4} className="ex-chart__tick">
            {x0}
          </text>
          <text x={w - M.r} y={H - 4} textAnchor="end" className="ex-chart__tick">
            {x1} {unit}
          </text>
          {marks.map((m) => (
            <g key={m.label}>
              <line x1={sx(m.x)} x2={sx(m.x)} y1={M.t} y2={H - M.b} className="ex-chart__mark" />
            </g>
          ))}
          {deployed != null && <line x1={sx(deployed)} x2={sx(deployed)} y1={M.t} y2={H - M.b} className="ex-chart__deployed" />}
          <polyline points={path} className="ex-chart__line" />
          <line x1={sx(cursor)} x2={sx(cursor)} y1={M.t} y2={H - M.b} className="ex-chart__cursor" />
          {ci >= 0 && <circle cx={sx(cursor)} cy={sy(ys[ci])} r={3.5} className="ex-chart__dot" />}
        </svg>
      )}
    </figure>
  )
}

// --------------------------------------------------------------------------- a saved tensor
function TensorView({ name, tensor, max, side }: { name: string; tensor: ReplayTensor; max: number; side: number }) {
  const { t } = useI18n()
  const lay = tensorLayout(tensor, side)
  const data = useMemo(() => Float32Array.from(tensor.data), [tensor])
  return (
    <div className="ex-block">
      <h4 className="ex-block__title">
        <code>{name}</code> <span className="num ex-muted">{shape(tensor.shape)}</span>
      </h4>
      {tensor.note && <p className="ex-hint">{t('ex.replay.tensorNote', { note: tensor.note })}</p>}
      {lay.kind === 'columns' && (
        <div className="ex-thumbs ex-thumbs--maps">
          {Array.from({ length: lay.cols }, (_, j) => (
            <figure key={j} className="ex-thumb">
              <HeatMap values={column(data, lay.rows, lay.cols, j)} width={side} height={side} mode="diverging" max={max} label={t('ex.replay.colAria', { name, j })} />
              <figcaption className="num">[:, {j}]</figcaption>
            </figure>
          ))}
        </div>
      )}
      {lay.kind === 'kernels' && (
        <div className="ex-thumbs ex-thumbs--kernels">
          {Array.from({ length: lay.cout * lay.cin }, (_, i) => (
            <figure key={i} className="ex-thumb">
              <HeatMap values={data.subarray(i * lay.kh * lay.kw, (i + 1) * lay.kh * lay.kw)} width={lay.kw} height={lay.kh} mode="diverging" max={max} label={`${name}[${Math.floor(i / lay.cin)}, ${i % lay.cin}]`} />
              <figcaption className="num">
                [{Math.floor(i / lay.cin)},{i % lay.cin}]
              </figcaption>
            </figure>
          ))}
        </div>
      )}
      {lay.kind === 'matrix' && <HeatMap values={data} width={lay.w} height={lay.h} mode="diverging" max={max} label={name} className="ex-replay__matrix" />}
      {lay.kind === 'vector' && <HeatMap values={data} width={lay.n} height={1} mode="diverging" max={max} label={name} style={{ aspectRatio: 'auto', height: 18 }} />}
      <p className="ex-hint">{t('ex.replay.scaleNote')}</p>
    </div>
  )
}
