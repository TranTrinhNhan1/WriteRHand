import { useState, type ReactNode } from 'react'
import { useI18n } from '../../../i18n'
import { usePersisted } from '../neural/ui'

/** Set once this browser failed to give a WebGL context: every 3D view then shows its 2D form. */
let webglFailed = false

/** 3D view of a network with its 2D form as the fallback (no WebGL) and as a choice. */
export function NetSwitch({ three, flat }: { three: (onUnavailable: () => void) => ReactNode; flat: () => ReactNode }) {
  const { t } = useI18n()
  const [mode, setMode] = usePersisted<'3d' | '2d'>('net.view', '3d')
  const [failed, setFailed] = useState(webglFailed)
  const fail = () => {
    webglFailed = true
    setFailed(true)
  }
  const show = failed ? '2d' : mode
  return (
    <div className="ex-netswitch" data-view={show}>
      {!failed && (
        <div className="ex-netswitch__tabs" role="group" aria-label={t('ex.three.mode')}>
          {(['3d', '2d'] as const).map((m) => (
            <button key={m} type="button" className="ex-netswitch__btn" aria-pressed={show === m} onClick={() => setMode(m)}>
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      )}
      {show === '3d' ? three(fail) : flat()}
    </div>
  )
}
