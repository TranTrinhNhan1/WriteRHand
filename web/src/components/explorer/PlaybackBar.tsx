import { ArrowCounterClockwise, Pause, Play, SkipBack, SkipForward } from '@phosphor-icons/react'
import type { CSSProperties } from 'react'
import { useI18n } from '../../i18n'
import { SPEEDS, type Playback } from '../../engine/react'

interface Props {
  pb: Playback
  /** Short description of the current step, shown next to the scrubber (e.g. "layer 2 of 5"). */
  caption?: string
  /** Unit name for the step counter, e.g. "layers". */
  unit?: string
}

/** Play / pause / step back / step forward / replay / speed / scrubber for any step-based animation. */
export function PlaybackBar({ pb, caption, unit }: Props) {
  const { t } = useI18n()
  const pct = pb.steps > 0 ? (100 * pb.step) / pb.steps : 100
  return (
    <div className="ex-playback" role="group" aria-label={t('ex.pb.label')}>
      <div className="ex-playback__btns">
        <button type="button" className="ex-pbtn" onClick={pb.prev} disabled={pb.step === 0} aria-label={t('ex.pb.prev')} title={t('ex.pb.prev')}>
          <SkipBack size={16} aria-hidden="true" />
        </button>
        <button
          type="button"
          className="ex-pbtn ex-pbtn--main"
          onClick={pb.toggle}
          aria-label={pb.playing ? t('ex.pb.pause') : t('ex.pb.play')}
          title={pb.playing ? t('ex.pb.pause') : t('ex.pb.play')}
        >
          {pb.playing ? <Pause size={16} weight="fill" aria-hidden="true" /> : <Play size={16} weight="fill" aria-hidden="true" />}
        </button>
        <button type="button" className="ex-pbtn" onClick={pb.next} disabled={pb.step >= pb.steps} aria-label={t('ex.pb.next')} title={t('ex.pb.next')}>
          <SkipForward size={16} aria-hidden="true" />
        </button>
        <button type="button" className="ex-pbtn" onClick={pb.replay} aria-label={t('ex.pb.replay')} title={t('ex.pb.replay')}>
          <ArrowCounterClockwise size={16} aria-hidden="true" />
        </button>
      </div>
      <input
        className="ex-playback__scrub"
        type="range"
        min={0}
        max={pb.steps}
        step={1}
        value={pb.step}
        style={{ '--p': `${pct}%` } as CSSProperties}
        onChange={(e) => pb.seek(Number(e.target.value))}
        aria-label={t('ex.pb.scrub')}
        aria-valuetext={`${pb.step} / ${pb.steps}${unit ? ` ${unit}` : ''}`}
      />
      <span className="num ex-playback__count">
        {pb.step}/{pb.steps}
      </span>
      <label className="ex-playback__speed">
        <span className="visually-hidden">{t('ex.pb.speed')}</span>
        <select value={pb.speed} onChange={(e) => pb.setSpeed(Number(e.target.value))}>
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
      {caption && <span className="ex-playback__caption">{caption}</span>}
    </div>
  )
}
