import { useState } from 'react'
import '../styles/results.css'
import '../styles/models.css'
import { Segmented } from '../components/Segmented'
import { Tex } from '../components/Tex'
import { modelsOf, type ModelKind } from '../engine/contract'
import { useBundle } from '../engine/react'
import { useI18n } from '../i18n'
import { parameterCount, summarizeLayers } from '../lib/architecture'
import { useChoice } from '../lib/choice'
import { bytes, int, paramName, paramValue, pct } from '../lib/format'
import { datasetIdsOf, downloadBytes } from '../lib/selection'
import { MODEL_TEX } from './modelContent'

const HIDDEN_PARAMS = new Set(['verbose', 'dtype'])

function Architecture({ dataset, id }: { dataset: string; id: ModelKind }) {
  const { t } = useI18n()
  const { manifest } = useBundle()
  const entry = manifest.datasets[dataset].models[id]!
  const layers = summarizeLayers(entry.trace)
  return (
    <figure className="arch">
      <figcaption className="caption">{t('models.arch.caption', { dataset: manifest.datasets[dataset].title })}</figcaption>
      <ol className="arch__chain">
        {layers.map((l) => (
          <li key={l.id} className={`arch__node arch__node--${l.op}`}>
            <span className="arch__op">{t(`models.op.${l.op}`)}</span>
            {l.detail && <span className="arch__detail num">{l.detail}</span>}
            <span className="arch__shape num">{l.shape}</span>
            {l.params != null && <span className="arch__params num">{t('models.arch.params', { n: int(l.params) })}</span>}
          </li>
        ))}
      </ol>
    </figure>
  )
}

function ModelArticle({ dataset, id }: { dataset: string; id: ModelKind }) {
  const { t, tl } = useI18n()
  const { manifest, results } = useBundle()
  const ds = manifest.datasets[dataset]
  const entry = ds.models[id]!
  const res = results[dataset]?.models.find((m) => m.id === id)
  const params = Object.entries(entry.params ?? res?.params ?? {}).filter(([k]) => !HIDDEN_PARAMS.has(k))
  const test = res?.splits?.test
  const v = entry.validation
  const captions = tl(`models.content.${id}.eq`)
  const nParams = parameterCount(entry.trace)
  const out = entry.trace.output

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
            <dt>{t('models.testAcc', { dataset: ds.title })}</dt>
            <dd className="num">{pct(test?.accuracy, 2)}</dd>
          </div>
          <div>
            <dt>{t('models.nParams')}</dt>
            <dd className="num">{nParams != null ? int(nParams) : t(id === 'adaboost' ? 'models.nParamsTrees' : 'common.na', { n: int(entry.trace.layers.find((l) => l.op === 'tree_ensemble')?.n_estimators) })}</dd>
          </div>
          <div>
            <dt>{t('models.size')}</dt>
            <dd className="num">{bytes(downloadBytes(entry))}</dd>
          </div>
          <div>
            <dt>{t('models.output')}</dt>
            <dd>{out.kind === 'vote_share' ? t('models.outVote') : t('models.outProb')}</dd>
          </div>
        </dl>
      </header>

      <Architecture dataset={dataset} id={id} />

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
              <p className="label">{t('models.params', { dataset: ds.title })}</p>
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
          {res && (res.best_epoch || res.stopped_epoch) ? (
            <p className="caption">
              {t('models.earlyStop', { stopped: res.stopped_epoch ?? res.best_epoch ?? 0, best: res.best_epoch ?? res.stopped_epoch ?? 0 })}
            </p>
          ) : null}
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
            {id === 'cnn' && (
              <>
                <div>
                  <dt>{t('models.notUsed')}</dt>
                  <dd>{t('models.content.cnn.notUsed')}</dd>
                </div>
                <div>
                  <dt>{t('models.verified')}</dt>
                  <dd>{t('models.content.cnn.verified')}</dd>
                </div>
              </>
            )}
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
                      diff: v.max_abs_prob_diff === 0 ? '0' : v.max_abs_prob_diff.toExponential(1),
                    })}
                  </>
                ) : null}
                {v && !v.ok ? <strong className="impl__warn"> {t('models.checkFailed')}</strong> : null}
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
  const { manifest } = useBundle()
  const { dataset, setDataset } = useChoice()
  const ds = manifest.datasets[dataset]
  const ids = modelsOf(ds)
  const [active, setActive] = useState<ModelKind | null>(null)
  const dsIds = datasetIdsOf(manifest)

  return (
    <div className="models sheet">
      <header className="page-head">
        <p className="eyebrow">{t('nav.models')}</p>
        <h1 className="display">{t('models.title', { n: ids.length })}</h1>
        <p className="lede">{t('models.lede')}</p>
        {dsIds.length > 1 && (
          <Segmented
            legend={t('models.datasetLegend')}
            value={dataset}
            onChange={setDataset}
            options={dsIds.map((id) => ({ value: id, label: manifest.datasets[id].title }))}
          />
        )}
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
        <ModelArticle key={`${dataset}-${id}`} dataset={dataset} id={id} />
      ))}
    </div>
  )
}
