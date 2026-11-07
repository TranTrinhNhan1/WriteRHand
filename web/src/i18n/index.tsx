import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { en, type Dict } from './en'
import { vi } from './vi'

export type Lang = 'vi' | 'en'
export const LANGS: Lang[] = ['vi', 'en']
export const DEFAULT_LANG: Lang = 'vi'
const STORAGE_KEY = 'writerhand.lang'

export const DICTS: Record<Lang, Dict> = { en, vi }

type Paths<T, P extends string = ''> = T extends string
  ? P
  : T extends readonly string[]
    ? never
    : { [K in keyof T & string]: Paths<T[K], P extends '' ? K : `${P}.${K}`> }[keyof T & string]

type ListPaths<T, P extends string = ''> = T extends readonly string[]
  ? P
  : T extends string
    ? never
    : { [K in keyof T & string]: ListPaths<T[K], P extends '' ? K : `${P}.${K}`> }[keyof T & string]

export type Key = Paths<Dict>
export type ListKey = ListPaths<Dict>
export type Params = Record<string, string | number>

function lookup(dict: Dict, key: string): unknown {
  let cur: unknown = dict
  for (const part of key.split('.')) {
    if (cur == null || typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[part]
  }
  return cur
}

function fill(template: string, params?: Params): string {
  if (!params) return template
  return template.replace(/\{(\w+)\}/g, (m, name: string) => (name in params ? String(params[name]) : m))
}

/** Plain translation, usable outside React (for example in tests or error mapping). */
export function translate(lang: Lang, key: Key, params?: Params): string {
  const v = lookup(DICTS[lang], key)
  return typeof v === 'string' ? fill(v, params) : key
}

export function translateList(lang: Lang, key: ListKey): string[] {
  const v = lookup(DICTS[lang], key)
  return Array.isArray(v) ? (v as string[]) : []
}

export function readStoredLang(): Lang | null {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY)
    return v === 'vi' || v === 'en' ? v : null
  } catch {
    return null
  }
}

function storeLang(lang: Lang) {
  try {
    window.localStorage.setItem(STORAGE_KEY, lang)
  } catch {
    /* private window or blocked storage: the choice just won't be remembered */
  }
}

/**
 * The active language is also kept at module level so plain helpers (number and date formatting) can read it.
 * The provider writes it during render, before any child renders, and every language-aware component reads the
 * context, so a switch always re-renders them.
 */
let active: Lang = DEFAULT_LANG
export function activeLang(): Lang {
  return active
}

interface I18n {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: Key, params?: Params) => string
  tl: (key: ListKey) => string[]
}

const Ctx = createContext<I18n | null>(null)

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => readStoredLang() ?? DEFAULT_LANG)
  active = lang

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const setLang = useCallback((l: Lang) => {
    storeLang(l)
    setLangState(l)
  }, [])

  const value = useMemo<I18n>(
    () => ({
      lang,
      setLang,
      t: (key, params) => translate(lang, key, params),
      tl: (key) => translateList(lang, key),
    }),
    [lang, setLang],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useI18n(): I18n {
  const v = useContext(Ctx)
  if (!v) throw new Error('useI18n needs a LangProvider above it.')
  return v
}
