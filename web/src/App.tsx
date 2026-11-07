import { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react'
import { LANGS, useI18n, type Key } from './i18n'
import { ModelStoreContext, SiteDataContext } from './lib/context'
import { MODEL_BASE_URL, HF_URL, REPO_URL } from './lib/config'
import { loadSiteData } from './lib/data'
import { CodedError } from './lib/errors'
import { date } from './lib/format'
import { useDocumentTitle } from './lib/hooks'
import { ModelStore } from './lib/modelStore'
import { ROUTES, useRoute, type Route } from './lib/route'
import type { SiteData } from './lib/types'
import Recognise from './pages/Recognise'

const Models = lazy(() => import('./pages/Models'))
const Results = lazy(() => import('./pages/Results'))
const About = lazy(() => import('./pages/About'))

const TITLE_KEY: Record<Route, Key> = {
  recognise: 'meta.title.recognise',
  models: 'meta.title.models',
  results: 'meta.title.results',
  about: 'meta.title.about',
}

function PageSkeleton() {
  const { t } = useI18n()
  return (
    <div className="page-skeleton" aria-busy="true" aria-label={t('common.loadingPage')}>
      <div className="skeleton skeleton--title" />
      <div className="skeleton skeleton--line" />
      <div className="skeleton skeleton--line skeleton--short" />
      <div className="skeleton skeleton--block" />
    </div>
  )
}

function LangSwitch() {
  const { t, lang, setLang } = useI18n()
  return (
    <div className="langswitch" role="group" aria-label={t('nav.language')} style={{ '--i': LANGS.indexOf(lang) } as React.CSSProperties}>
      <span className="langswitch__thumb" aria-hidden="true" />
      {LANGS.map((l) => (
        <button key={l} type="button" lang={l} aria-pressed={l === lang} onClick={() => setLang(l)}>
          {l.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

function Header({ route }: { route: Route }) {
  const { t } = useI18n()
  return (
    <header className="site-header">
      <div className="wrap site-header__inner">
        <a className="wordmark" href="#/" aria-label={t('nav.home')}>
          <span className="wordmark__mark" aria-hidden="true" />
          WriteRHand
        </a>
        <nav aria-label={t('nav.label')} className="site-nav">
          <ul className="nav">
            {ROUTES.map((r) => (
              <li key={r.id}>
                <a href={r.hash} aria-current={r.id === route ? 'page' : undefined}>
                  {t(`nav.${r.id}`)}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <LangSwitch />
      </div>
    </header>
  )
}

function Footer({ data }: { data: SiteData | null }) {
  const { t } = useI18n()
  const git = data?.results.git
  const bits = [
    data?.manifest.version ? t('footer.bundle', { version: data.manifest.version }) : null,
    git?.short ? `${t('footer.commit', { sha: git.short })}${git.dirty ? ` (${t('footer.dirty')})` : ''}` : null,
    data?.results.generated ? t('footer.generated', { date: date(data.results.generated) }) : null,
  ].filter(Boolean)
  const [before, rest] = t('footer.about', { github: '\u0000', hf: '\u0001' }).split('\u0000')
  const [between, after] = rest.split('\u0001')
  return (
    <footer className="site-footer">
      <div className="wrap site-footer__inner">
        <div className="site-footer__word" aria-hidden="true" />
        <p className="site-footer__about">
          {before}
          <a href={REPO_URL}>{t('footer.github')}</a>
          {between}
          <a href={HF_URL}>{t('footer.hf')}</a>
          {after}
        </p>
        {bits.length > 0 && <p className="num site-footer__meta">{bits.join(' · ')}</p>}
      </div>
    </footer>
  )
}

export default function App() {
  const { t, lang } = useI18n()
  const route = useRoute()
  const [data, setData] = useState<SiteData | null>(null)
  const [error, setError] = useState<CodedError | null>(null)
  const [attempt, setAttempt] = useState(0)
  const mainRef = useRef<HTMLElement>(null)
  useDocumentTitle(t(TITLE_KEY[route]))

  useEffect(() => {
    const meta = document.querySelector('meta[name="description"]')
    meta?.setAttribute('content', t('meta.description'))
  }, [t, lang])

  useEffect(() => {
    let cancelled = false
    setError(null)
    loadSiteData().then(
      (d) => !cancelled && setData(d),
      (e: unknown) => !cancelled && setError(e instanceof CodedError ? e : new CodedError('fatal.reach')),
    )
    return () => {
      cancelled = true
    }
  }, [attempt])

  const store = useMemo(() => (data ? new ModelStore(data.manifest, MODEL_BASE_URL) : null), [data])

  // Hash navigation does not reset scroll or focus on its own.
  const first = useRef(true)
  useEffect(() => {
    if (first.current) {
      first.current = false
      return
    }
    window.scrollTo({ top: 0 })
    mainRef.current?.focus({ preventScroll: true })
  }, [route])

  let body: React.ReactNode
  if (error) {
    body = (
      <div className="wrap">
        <div className="fatal sheet" role="alert">
          <h1 className="h2">{t('fatal.title')}</h1>
          <p>{t(error.key, error.params)}</p>
          <button type="button" className="btn btn--primary" onClick={() => setAttempt((a) => a + 1)}>
            {t('common.retry')}
          </button>
        </div>
      </div>
    )
  } else if (!data || !store) {
    body = (
      <div className="wrap">
        <PageSkeleton />
      </div>
    )
  } else {
    body = (
      <SiteDataContext.Provider value={data}>
        <ModelStoreContext.Provider value={store}>
          <Suspense
            fallback={
              <div className="wrap">
                <PageSkeleton />
              </div>
            }
          >
            <div className="wrap">
              {route === 'recognise' && <Recognise />}
              {route === 'models' && <Models />}
              {route === 'results' && <Results />}
              {route === 'about' && <About />}
            </div>
          </Suspense>
        </ModelStoreContext.Provider>
      </SiteDataContext.Provider>
    )
  }

  return (
    <>
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          e.preventDefault()
          mainRef.current?.focus()
        }}
      >
        {t('nav.skip')}
      </a>
      <Header route={route} />
      <main id="main" ref={mainRef} tabIndex={-1}>
        {body}
      </main>
      <Footer data={data} />
    </>
  )
}
