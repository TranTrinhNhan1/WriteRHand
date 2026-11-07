import { useId, type CSSProperties } from 'react'
import { useI18n } from '../i18n'
import { bytes } from '../lib/format'
import type { StatusMap } from '../lib/modelStore'
import type { Manifest, ModelId } from '../lib/types'

interface Props {
  ids: ModelId[]
  selected: ModelId
  statuses: StatusMap
  manifest: Manifest
  onSelect: (id: ModelId) => void
}

/** One row per classifier. A highlight slides to the chosen row, and each row says whether its model is loaded. */
export function ModelPicker({ ids, selected, statuses, manifest, onSelect }: Props) {
  const { t } = useI18n()
  const name = useId()
  const index = Math.max(0, ids.indexOf(selected))
  const style = { '--i': index, '--n': ids.length } as CSSProperties

  const statusText = (id: ModelId): string => {
    const s = statuses[id]
    const size = bytes(manifest.models[id]?.size_bytes)
    if (!s || s.state === 'idle') return t('picker.idle', { size })
    if (s.state === 'loading') return t('picker.loading')
    if (s.state === 'error') return t('picker.error')
    return t('picker.ready', { size })
  }

  return (
    <fieldset className="picker" style={style}>
      <legend className="visually-hidden">{t('picker.legend')}</legend>
      <div className="picker__rows">
        <span className="picker__thumb" aria-hidden="true" />
        {ids.map((id) => {
          const s = statuses[id]
          return (
            <label key={id} className={`picker__opt${s?.state === 'error' ? ' is-error' : ''}`}>
              <input type="radio" name={name} value={id} checked={selected === id} onChange={() => onSelect(id)} />
              <span className="picker__name">{t(`model.${id}`)}</span>
              <span className="picker__meta">
                <span className={`dot dot--${s?.state ?? 'idle'}`} aria-hidden="true" />
                {statusText(id)}
              </span>
            </label>
          )
        })}
      </div>
    </fieldset>
  )
}
