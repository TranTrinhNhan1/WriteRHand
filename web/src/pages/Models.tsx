import { useState } from 'react'
import '../styles/results.css'
import '../styles/models.css'
import { Tex } from '../components/Tex'
import { useI18n } from '../i18n'
import { useSiteData } from '../lib/context'
import { bytes, int, paramName, paramValue, pct } from '../lib/format'
import { MODEL_IDS, type ModelId } from '../lib/types'
import { MODEL_TEX } from './modelContent'

const HIDDEN_PARAMS = new Set(['verbose', 'dtype'])

function ModelArticle({ id }: { id: ModelId }) {
  const { t, tl } = useI18n()
  const { manifest, results } = useSiteData()
  const entry = manifest.models[id]
  const res = results.models.find((m) => m.id === id)
  const params = Object.entries(res?.params ?? entry?.params ?? {}).filter(([k]) => !HIDDEN_PARAMS.has(k))
  const test = res?.splits?.test
  const v = entry?.validation
  const captions = tl(`models.content.${id}.eq`)

  return (
    <article id={`model-${id}`} className="model" aria-labelledby={`model-${id}-h`}>
      <header className="model__head">
        <h2 id={`model-${id}-h`} className="h2">
          {t(`model.${id}`)}
        </h2>
        <p className="lede">{t(`models.content.${id}.tagline`)}</p>
        <dl className="facts facts--inline">
          <div>
            <dt>{t('models.family')}</dt>
            <dd>{t(`models.content.${id}.family`)}</dd>
          </div>
          <div>
            <dt>{t('models.testAcc')}</dt>
            <dd className="num">{pct(test?.accuracy, 2)}</dd>
          </div>
          <div>
            <dt>{t('models.size')}</dt>
            <dd className="num">{bytes(entry?.size_bytes)}</dd>
          </div>
        </dl>
      </header>

      <div className="model__body">
        <div className="model__col">
          <h3 className="h3">{t('models.how')}</h3>
          {tl(`models.content.${id}.how`).map((p) => (
            <p key={p}>{p}</p>
          ))}
          <div className="equations">
            {MODEL_TEX[id].map((lines, i) => (
              <figure key={captions[i] ?? i} className="equation">
                <figcaption className="caption">{captions[i]}</figcaption>
                {lines.map((line) => (
                  <Tex key={line} tex={line} block />
                ))}
              </figure>
            ))}
          </div>
        </div>

        <div className="model__col">
          <h3 className="h3">{t('models.training')}</h3>
          {tl(`models.content.${id}.training`).map((p) => (
            <p key={p}>{p}</p>
          ))}
          {params.length > 0 && (
            <>
              <p className="label">{t('models.params')}</p>
              <dl className="params params--grid">
                {params.map(([k, val]) => (
                  <div key={k}>
                    <dt>{paramName(k)}</dt>
                    <dd className="num">{paramValue(val)}</dd>
                  </div>
                ))}
              </dl>
            </>
          )}
          {res && (res.best_epoch || res.stopped_epoch) && (
            <p className="caption">
              {t('models.earlyStop', { stopped: res.stopped_epoch ?? res.best_epoch ?? 0, best: res.best_epoch ?? res.stopped_epoch ?? 0 })}
            </p>
          )}
        </div>

        <div className="model__pros">
          <div>
            <h3 className="h3">{t('models.strengths')}</h3>
            <ul className="bullets">
              {tl(`models.content.${id}.strengths`).map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="h3">{t('models.limits')}</h3>
            <ul className="bullets">
              {tl(`models.content.${id}.limits`).map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="model__col model__impl">
          <h3 className="h3">{t('models.ours')}</h3>
          <dl className="impl">
            <div>
              <dt>{t('models.written')}</dt>
              <dd>{t(`models.content.${id}.scratch`)}</dd>
            </div>
            <div>
              <dt>{t('models.library')}</dt>
              <dd>{t(`models.content.${id}.library`)}</dd>
            </div>
            <div>
              <dt>{t('models.here')}</dt>
              <dd>
                {t(`models.content.${id}.browser`)}
                {v?.n_checked ? (
                  <>
                    {' '}
                    {t('models.checked', {
                      n: int(v.n_checked),
                      agree: pct(v.argmax_agreement, 1),
                      diff: v.max_abs_prob_diff == null ? t('common.na') : v.max_abs_prob_diff === 0 ? '0' : v.max_abs_prob_diff.toExponential(1),
                    })}
                  </>
                ) : null}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </article>
  )
}

export default function Models() {
  const { t } = useI18n()
  const { manifest } = useSiteData()
  const ids = MODEL_IDS.filter((id) => manifest.models[id])
  const [active, setActive] = useState<ModelId>(ids[0])

  return (
    <div className="models sheet">
      <header className="page-head">
        <p className="eyebrow">{t('nav.models')}</p>
        <h1 className="display">{t('models.title')}</h1>
        <p className="lede">{t('models.lede')}</p>
        <nav aria-label={t('models.jump')} className="jump">
          <ul>
            {ids.map((id) => (
              <li key={id}>
                <button
                  type="button"
                  className="btn btn--quiet"
                  aria-current={active === id ? 'true' : undefined}
                  onClick={() => {
                    setActive(id)
                    document.getElementById(`model-${id}`)?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' })
                  }}
                >
                  {t(`model.${id}`)}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </header>
      {ids.map((id) => (
        <ModelArticle key={id} id={id} />
      ))}
    </div>
  )
}
