import '../styles/results.css'
import '../styles/models.css'
import '../styles/about.css'
import { useBundle } from '../engine/react'
import { useI18n } from '../i18n'
import { useChoice } from '../lib/choice'
import { REPO_URL, hfRepoUrl, hfRevisionUrl, shortHash } from '../lib/config'
import { int } from '../lib/format'
import { datasetIdsOf } from '../lib/selection'
import { DatasetsSection } from './AboutDatasets'

export default function About() {
  const { t, tl } = useI18n()
  const { manifest, lock } = useBundle()
  const ids = datasetIdsOf(manifest)
  const { dataset } = useChoice()
  const ds = manifest.datasets[dataset]
  const d = ds.dataset
  const revUrl = hfRevisionUrl(lock)
  const flow = tl('about.flow.steps')

  return (
    <div className="about sheet">
      <header className="page-head">
        <h1 className="display">{t('about.title')}</h1>
        <p className="lede">{t('about.lede')}</p>
        <ul className="linkrow">
          <li>
            <a className="btn btn--quiet" href={REPO_URL}>
              {t('about.repo')}
            </a>
          </li>
          <li>
            <a className="btn btn--quiet" href={revUrl ?? hfRepoUrl(lock)}>
              {t('about.hf')}
            </a>
          </li>
        </ul>
      </header>

      <div className="prose">
        <section aria-labelledby="a-scope">
          <h2 id="a-scope" className="h2">
            {t('about.scope.title')}
          </h2>
          <ul className="bullets">
            {tl('about.scope.items').map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>

        <DatasetsSection />

        <section aria-labelledby="a-split">
          <h2 id="a-split" className="h2">
            {t('about.split.title')}
          </h2>
          <p>{t('about.split.text')}</p>
          <dl className="kv">
            <div>
              <dt>{t('about.pre.dataset')}</dt>
              <dd>{ds.title}</dd>
            </div>
            <div>
              <dt>{t('about.split.train')}</dt>
              <dd className="num">{t('about.split.trainText', { n: int(d?.n_train) })}</dd>
            </div>
            <div>
              <dt>{t('about.split.val')}</dt>
              <dd className="num">{t('about.split.valText', { n: int(d?.n_val) })}</dd>
            </div>
            <div>
              <dt>{t('about.split.test')}</dt>
              <dd className="num">{t('about.split.testText', { n: int(d?.n_test) })}</dd>
            </div>
            {d?.seed != null && (
              <div>
                <dt>{t('about.split.seed')}</dt>
                <dd className="num">{d.seed}</dd>
              </div>
            )}
            {d?.split_fingerprint && (
              <div>
                <dt>{t('about.split.fingerprint')}</dt>
                <dd className="num">{d.split_fingerprint}</dd>
              </div>
            )}
          </dl>
          <p>{t('about.split.note')}</p>
        </section>

        <section aria-labelledby="a-pre">
          <h2 id="a-pre" className="h2">
            {t('about.pre.title')}
          </h2>
          <p>{t('about.pre.text1')}</p>
          <p className="formula num">{manifest.datasets[manifest.default_dataset].preprocessing.formula ?? t('about.pre.formula')}</p>
          <div className="table-wrap" role="region" aria-label={t('about.pre.tableLabel')} tabIndex={0}>
            <table className="table num">
              <thead>
                <tr>
                  <th scope="col" className="is-left">
                    {t('about.pre.dataset')}
                  </th>
                  <th scope="col">{t('about.pre.mean')}</th>
                  <th scope="col">{t('about.pre.std')}</th>
                  <th scope="col">{t('about.pre.eps')}</th>
                </tr>
              </thead>
              <tbody>
                {ids.map((id) => {
                  const p = manifest.datasets[id].preprocessing
                  return (
                    <tr key={id}>
                      <th scope="row" className="is-left">
                        {manifest.datasets[id].title}
                      </th>
                      <td>{p.mean.toFixed(6)}</td>
                      <td>{p.std.toFixed(6)}</td>
                      <td>{p.eps}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p>{t('about.pre.text2')}</p>
        </section>

        <section aria-labelledby="a-vis">
          <h2 id="a-vis" className="h2">
            {t('about.vis.title')}
          </h2>
          {tl('about.vis.text').map((p) => (
            <p key={p}>{p}</p>
          ))}
        </section>

        <section aria-labelledby="a-flow">
          <h2 id="a-flow" className="h2">
            {t('about.flow.title')}
          </h2>
          <p>{t('about.flow.text')}</p>
          <ol className="flow">
            {flow.map((step, i) => (
              <li key={step}>
                <span className="flow__no num" aria-hidden="true">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
          <p>
            {lock && revUrl ? (
              <>
                {t('about.flow.pinned', { repo: lock.repo })}{' '}
                <a href={revUrl} className="num">
                  {shortHash(lock.revision, 12)}
                </a>
                {t('about.flow.pinnedEnd', { bundle: shortHash(manifest.bundle_id) })}
              </>
            ) : (
              t('about.flow.local', { bundle: shortHash(manifest.bundle_id) })
            )}
          </p>
        </section>

        <section aria-labelledby="a-edu">
          <h2 id="a-edu" className="h2">
            {t('about.why.title')}
          </h2>
          <p>{t('about.why.text')}</p>
        </section>
      </div>
    </div>
  )
}
