import '../styles/results.css'
import '../styles/about.css'
import '../styles/documents.css'
import { useEffect, useState, type ReactNode } from 'react'
import { ArrowSquareOut, DownloadSimple, FilePdf, GithubLogo, LockSimple } from '@phosphor-icons/react'
import { MODEL_LOCK } from '../engine/lock'
import { useI18n, type Key } from '../i18n'
import { REPO_PUBLIC, REPO_URL, hfRepoUrl, hfRevisionUrl } from '../lib/config'
import { docFileUrl, loadDocsMeta, type DocId, type DocsMeta } from '../lib/docs'
import { bytes } from '../lib/format'

type MetaState = { status: 'loading' } | { status: 'ok'; meta: DocsMeta } | { status: 'error' }

const COUNT_KEY: Record<DocId, Key> = { report: 'docs.pages', slides: 'docs.slidesCount' }

/** Opens in a new tab; external links never get access to this page. */
const NEW_TAB = { target: '_blank', rel: 'noopener noreferrer' } as const

function DocCard({ id, meta }: { id: DocId; meta: MetaState }) {
  const { t, lang } = useI18n()
  const url = docFileUrl(id)
  const file = meta.status === 'ok' ? meta.meta[id] : null
  const section = t(`docs.${id}.title`)
  const headingId = `doc-${id}`
  return (
    <section className="docsec" aria-labelledby={headingId}>
      <h2 id={headingId} className="h2">
        {section}
      </h2>
      <article className="doccard">
        <span className="doccard__icon" aria-hidden="true">
          <FilePdf size={26} weight="light" />
        </span>
        <div className="doccard__body">
          <h3 className="h3 doccard__name" lang={id === 'report' ? 'vi' : undefined}>
            {t(`docs.${id}.name`)}
          </h3>
          <p className="doccard__text">
            {t(`docs.${id}.text`)}
            {lang !== 'vi' ? ` ${t('docs.inVietnamese')}` : ''}
          </p>
          {meta.status === 'loading' ? (
            <p className="doccard__meta" aria-busy="true">
              <span className="visually-hidden">{t('docs.metaLoading')}</span>
              <span className="skeleton doccard__skel" aria-hidden="true" />
            </p>
          ) : file ? (
            <p className="doccard__meta num">
              <span data-pages={file.pages}>{t(COUNT_KEY[id], { n: file.pages })}</span>
              <span aria-hidden="true">·</span>
              <span>{bytes(file.bytes)}</span>
              <span aria-hidden="true">·</span>
              <span>{t('docs.pdf')}</span>
            </p>
          ) : (
            <p className="doccard__meta doccard__meta--error">{t('docs.metaError')}</p>
          )}
          <ul className="linkrow doccard__actions">
            <li>
              <a className="btn btn--primary" href={url} {...NEW_TAB} aria-label={t('docs.viewLabel', { doc: section })}>
                <ArrowSquareOut size={18} aria-hidden="true" />
                {t('docs.view')}
              </a>
            </li>
            <li>
              <a className="btn btn--quiet" href={url} download={`WriteRHand-${id}.pdf`} aria-label={t('docs.downloadLabel', { doc: section })}>
                <DownloadSimple size={18} aria-hidden="true" />
                {t('docs.download')}
              </a>
            </li>
          </ul>
        </div>
      </article>
    </section>
  )
}

/** A ruled-list link: icon, underlined label (and an optional repository path), and a tag while the repository is private. */
function LinkRow({ href, icon, label, path, github = false }: { href: string; icon: ReactNode; label: string; path?: string; github?: boolean }) {
  const { t } = useI18n()
  return (
    <a className="doclink" href={href} {...NEW_TAB}>
      {icon}
      <span className="doclink__main">
        <span className="doclink__label">{label}</span>
        {path && <code className="doclink__path">{path}</code>}
      </span>
      {github && !REPO_PUBLIC && (
        <span className="doclink__tag">
          <LockSimple size={12} weight="bold" aria-hidden="true" />
          {t('docs.code.private')}
        </span>
      )}
    </a>
  )
}

const ghIcon = <GithubLogo size={18} aria-hidden="true" />
const extIcon = <ArrowSquareOut size={18} aria-hidden="true" />

export default function Documents() {
  const { t } = useI18n()
  const [meta, setMeta] = useState<MetaState>({ status: 'loading' })
  const lock = MODEL_LOCK
  const revUrl = hfRevisionUrl(lock)

  useEffect(() => {
    let cancelled = false
    loadDocsMeta().then(
      (m) => !cancelled && setMeta(m ? { status: 'ok', meta: m } : { status: 'error' }),
      () => !cancelled && setMeta({ status: 'error' }),
    )
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="docs sheet">
      <header className="page-head">
        <p className="eyebrow">{t('nav.documents')}</p>
        <h1 className="display">{t('docs.title')}</h1>
        <p className="lede">{t('docs.lede')}</p>
      </header>

      <div className="prose docs__body">
        <DocCard id="report" meta={meta} />
        <DocCard id="slides" meta={meta} />

        <section aria-labelledby="doc-code">
          <h2 id="doc-code" className="h2">
            {t('docs.code.title')}
          </h2>
          <p>{t('docs.code.text')}</p>
          {!REPO_PUBLIC && (
            <p className="docnotice" role="note">
              <LockSimple size={18} aria-hidden="true" />
              <span>{t('docs.code.privateNote')}</span>
            </p>
          )}
          <ul className="doclinks">
            <li>
              <LinkRow github href={REPO_URL} icon={ghIcon} label={t('docs.code.repo')} />
            </li>
            <li>
              <LinkRow github href={`${REPO_URL}/blob/main/README.md`} icon={ghIcon} label={t('docs.code.readme')} path="README.md" />
            </li>
            <li>
              <LinkRow
                github
                href={`${REPO_URL}/blob/main/notebooks/reproduce_results.ipynb`}
                icon={ghIcon}
                label={t('docs.code.notebook')}
                path="notebooks/reproduce_results.ipynb"
              />
            </li>
          </ul>
        </section>

        <section aria-labelledby="doc-exp">
          <h2 id="doc-exp" className="h2">
            {t('docs.exp.title')}
          </h2>
          <p>{t('docs.exp.models')}</p>
          <ul className="doclinks">
            <li>
              <LinkRow href={hfRepoUrl(lock)} icon={extIcon} label={t('docs.exp.hf')} path={lock?.repo} />
            </li>
            <li>
              {lock && revUrl ? (
                <LinkRow href={revUrl} icon={extIcon} label={t('docs.exp.revision')} path={lock.revision} />
              ) : (
                <span className="doclink doclink--static">{t('docs.exp.noRevision')}</span>
              )}
            </li>
          </ul>
          <p>{t('docs.exp.tracking')}</p>
          <p>{t('docs.exp.resultsText')}</p>
          <ul className="doclinks">
            <li>
              <LinkRow github href={`${REPO_URL}/tree/main/results`} icon={ghIcon} label={t('docs.exp.results')} path="results/" />
            </li>
          </ul>
        </section>
      </div>
    </div>
  )
}
