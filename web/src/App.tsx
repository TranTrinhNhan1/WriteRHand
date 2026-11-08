import { Suspense, lazy, useEffect, useMemo, useRef, useState } from 'react'
import { GithubLogo, Moon, Sun } from '@phosphor-icons/react'
import { loadBundle, type Booted } from './engine/boot'
import { BundleContext, type Bundle } from './engine/react'
import { ModelRegistry } from './engine/store'
import { LANGS, useI18n, type Key } from './i18n'
import { ChoiceProvider } from './lib/choice'
import { REPO_URL, hfRepoUrl, hfRevisionUrl, shortHash } from './lib/config'
import { CodedError } from './lib/errors'
import { date } from './lib/format'
import { useDocumentTitle } from './lib/hooks'
import { loadOrt } from './lib/ortLoader'
import { sanitizeResults } from './lib/results'
import { useTheme } from './lib/theme'
import { ROUTES, useRoute, type Route } from './lib/route'
import Recognise from './pages/Recognise'

const Models = lazy(() => import('./pages/Models'))
const Results = lazy(() => import('./pages/Results'))
const About = lazy(() => import('./pages/About'))
const Documents = lazy(() => import('./pages/Documents'))

const TITLE_KEY: Record<Route, Key> = {
  recognise: 'meta.title.recognise',
  models: 'meta.title.models',
  results: 'meta.title.results',
  about: 'meta.title.about',
  documents: 'meta.title.documents',
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

function ThemeToggle() {
  const { t } = useI18n()
  const [theme, toggle] = useTheme()
  const label = theme === 'dark' ? t('nav.toLight') : t('nav.toDark')
  return (
    <button type="button" className="theme-toggle" onClick={toggle} aria-label={label} title={label}>
      {theme === 'dark' ? <Sun size={18} weight="regular" aria-hidden="true" /> : <Moon size={18} weight="regular" aria-hidden="true" />}
    </button>
  )
}

function GitHubLink() {
  const { t } = useI18n()
  const label = t('nav.github')
  return (
    <a className="theme-toggle header-github" href={REPO_URL} target="_blank" rel="noopener noreferrer" aria-label={label} title={label}>
      <GithubLogo size={18} weight="regular" aria-hidden="true" />
    </a>
  )
}

function Header({ route }: { route: Route }) {
  const { t } = useI18n()
  return (
    <header className="site-header">
      <div className="wrap site-header__inner">
        <a className="wordmark" href="#/" aria-label={t('nav.home')}>
          <span className="wordmark__mark" aria-hidden="true" />
          <span className="wordmark__text">WriteRHand</span>
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
        <div className="header-tools">
          <LangSwitch />
          <ThemeToggle />
          <GitHubLink />
        </div>
      </div>
    </header>
  )
}

function Footer({ bundle }: { bundle: Bundle | null }) {
  const { t } = useI18n()
  const lock = bundle?.lock ?? null
  const first = bundle ? bundle.results[bundle.manifest.default_dataset] : null
  const revUrl = hfRevisionUrl(lock)
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
          <a href={hfRepoUrl(lock)}>{t('footer.hf')}</a>
          {after}
        </p>
        {bundle && (
          <p className="num site-footer__meta">
            <span>{t('footer.bundleId', { id: shortHash(bundle.manifest.bundle_id) })}</span>
            {' · '}
            {lock && revUrl ? (
              <a href={revUrl} title={lock.revision}>
                {t('footer.revision', { rev: shortHash(lock.revision, 7) })}
              </a>
            ) : (
              <span>{t('footer.localBundle')}</span>
            )}
            {first?.git?.short ? (
              <>
                {' · '}
                <span>{`${t('footer.commit', { sha: first.git.short })}${first.git.dirty ? ` (${t('footer.dirty')})` : ''}`}</span>
              </>
            ) : null}
            {first?.generated ? (
              <>
                {' · '}
                <span>{t('footer.generated', { date: date(first.generated) })}</span>
              </>
            ) : null}
          </p>
        )}
      </div>
    </footer>
  )
}

export default function App() {
  const { t, lang } = useI18n()
  const route = useRoute()
  const [booted, setBooted] = useState<Booted | null>(null)
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
    loadBundle().then(
      (b) => !cancelled && setBooted(b),
      (e: unknown) => !cancelled && setError(e instanceof CodedError ? e : new CodedError('fatal.reach')),
    )
    return () => {
      cancelled = true
    }
  }, [attempt])

  const bundle = useMemo<Bundle | null>(() => {
    if (!booted) return null
    const results = Object.fromEntries(Object.entries(booted.results).map(([id, r]) => [id, sanitizeResults(r)]))
    return { ...booted, results, registry: new ModelRegistry(booted.manifest, booted.baseUrl, { ortLoader: loadOrt }) }
  }, [booted])

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

  const pageFallback = (
    <div className="wrap">
      <PageSkeleton />
    </div>
  )
  let body: React.ReactNode
  if (route === 'documents') {
    // the deliverables do not depend on the model data, so they stay reachable even if it fails to load
    body = (
      <Suspense fallback={pageFallback}>
        <div className="wrap">
          <Documents />
        </div>
      </Suspense>
    )
  } else if (error) {
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
  } else if (!bundle) {
    body = (
      <div className="wrap">
        <PageSkeleton />
      </div>
    )
  } else {
    body = (
      <BundleContext.Provider value={bundle}>
        <ChoiceProvider>
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
        </ChoiceProvider>
      </BundleContext.Provider>
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
      <Footer bundle={bundle} />
    </>
  )
}
