import '../styles/results.css'
import '../styles/models.css'
import '../styles/about.css'
import { useI18n } from '../i18n'
import { MODEL_BASE_URL, HF_URL, REPO_URL } from '../lib/config'
import { useSiteData } from '../lib/context'
import { date, int } from '../lib/format'

function host(url: string): string {
  try {
    return new URL(url, window.location.href).host
  } catch {
    return url
  }
}

export default function About() {
  const { t, tl } = useI18n()
  const { manifest, results } = useSiteData()
  const ds = results.dataset ?? manifest.dataset
  const pre = manifest.preprocessing
  const generated = results.generated
    ? t('about.run.generated', {
        date: date(results.generated),
        commit: results.git?.short ? t('about.run.commit', { sha: results.git.short }) : '',
      })
    : ''

  return (
    <div className="about sheet">
      <header className="page-head">
        <p className="eyebrow">{t('nav.about')}</p>
        <h1 className="display">{t('about.title')}</h1>
        <p className="lede">{t('about.lede')}</p>
        <ul className="linkrow">
          <li>
            <a className="btn btn--quiet" href={REPO_URL}>
              {t('about.repo')}
            </a>
          </li>
          <li>
            <a className="btn btn--quiet" href={HF_URL}>
              {t('about.hf')}
            </a>
          </li>
        </ul>
      </header>

      <div className="prose">
        <section aria-labelledby="a-data">
          <h2 id="a-data" className="h2">
            {t('about.data.title')}
          </h2>
          <p>{t('about.data.text')}</p>
        </section>

        <section aria-labelledby="a-split">
          <h2 id="a-split" className="h2">
            {t('about.split.title')}
          </h2>
          <p>{t('about.split.text')}</p>
          <dl className="kv">
            <div>
              <dt>{t('about.split.train')}</dt>
              <dd className="num">{t('about.split.trainText', { n: int(ds?.n_train) })}</dd>
            </div>
            <div>
              <dt>{t('about.split.val')}</dt>
              <dd className="num">{t('about.split.valText', { n: int(ds?.n_val) })}</dd>
            </div>
            <div>
              <dt>{t('about.split.test')}</dt>
              <dd className="num">{t('about.split.testText', { n: int(ds?.n_test) })}</dd>
            </div>
            {ds?.seed != null && (
              <div>
                <dt>{t('about.split.seed')}</dt>
                <dd className="num">{ds.seed}</dd>
              </div>
            )}
            {ds?.split_fingerprint && (
              <div>
                <dt>{t('about.split.fingerprint')}</dt>
                <dd className="num">{ds.split_fingerprint}</dd>
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
          <p className="formula num">{pre.formula ?? t('about.pre.formula')}</p>
          <dl className="kv">
            <div>
              <dt>{t('about.pre.mean')}</dt>
              <dd className="num">{pre.mean}</dd>
            </div>
            <div>
              <dt>{t('about.pre.std')}</dt>
              <dd className="num">{pre.std}</dd>
            </div>
            <div>
              <dt>{t('about.pre.eps')}</dt>
              <dd className="num">{pre.eps}</dd>
            </div>
          </dl>
          <p>{t('about.pre.text2')}</p>
        </section>

        <section aria-labelledby="a-edu">
          <h2 id="a-edu" className="h2">
            {t('about.why.title')}
          </h2>
          <p>{t('about.why.text')}</p>
        </section>

        <section aria-labelledby="a-lim">
          <h2 id="a-lim" className="h2">
            {t('about.limits.title')}
          </h2>
          <ul className="bullets">
            {tl('about.limits.items').map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="a-run">
          <h2 id="a-run" className="h2">
            {t('about.run.title')}
          </h2>
          <p>{t('about.run.text', { host: host(MODEL_BASE_URL), generated })}</p>
        </section>

        <section aria-labelledby="a-ai">
          <h2 id="a-ai" className="h2">
            {t('about.ai.title')}
          </h2>
          <p>{t('about.ai.text')}</p>
        </section>
      </div>
    </div>
  )
}
