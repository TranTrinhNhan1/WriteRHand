import { activeLang, translate, type Lang } from '../i18n'

const NA = '\u2013'
/** Narrow no-break space: groups thousands without a comma, in both languages. */
const GROUP = '\u202f'

function unit(key: 'min' | 's' | 'ms' | 'us' | 'lessThan'): string {
  return translate(activeLang(), `unit.${key}`)
}

export function pct(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? NA : `${(v * 100).toFixed(digits)}%`
}

export function num(v: number | null | undefined, digits = 3): string {
  return v == null || !Number.isFinite(v) ? NA : v.toFixed(digits)
}

/** Whole numbers; groups of three digits are separated by a narrow space once a number has five or more digits. */
export function int(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return NA
  const r = Math.round(v)
  const s = String(Math.abs(r))
  const grouped = s.length >= 5 ? s.replace(/\B(?=(\d{3})+(?!\d))/g, GROUP) : s
  return r < 0 ? `\u2212${grouped}` : grouped
}

export function bytes(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return NA
  if (v < 1024) return `${v} B`
  if (v < 1024 * 1024) return `${(v / 1024).toFixed(v < 10240 ? 1 : 0)} KB`
  return `${(v / 1024 / 1024).toFixed(1)} MB`
}

/** Seconds with sensible units: 0.02 s -> "20 ms", 75 s -> "1 min 15 s" ("1 phút 15 giây"). */
export function seconds(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return NA
  if (v < 1e-3) return `${(v * 1e6).toFixed(0)} ${unit('us')}`
  if (v < 1) return `${(v * 1e3).toFixed(v < 0.01 ? 1 : 0)} ${unit('ms')}`
  if (v < 60) return `${v.toFixed(v < 10 ? 2 : 1)} ${unit('s')}`
  const m = Math.floor(v / 60)
  const s = Math.round(v - m * 60)
  return `${m} ${unit('min')} ${s} ${unit('s')}`
}

export function ms(v: number | null | undefined, digits?: number): string {
  if (v == null || !Number.isFinite(v)) return NA
  const d = digits ?? (v < 1 ? 3 : v < 10 ? 2 : 1)
  return `${v.toFixed(d)} ${unit('ms')}`
}

/** Wall-clock timings from performance.now() are coarse (0.1 ms), so do not print more digits than that. */
export function latency(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return NA
  if (v < 0.05) return unit('lessThan')
  return `${v.toFixed(v < 10 ? 1 : 0)} ${unit('ms')}`
}

export function date(iso: string | undefined, lang: Lang = activeLang()): string {
  if (!iso) return NA
  const d = new Date(iso)
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString(lang === 'vi' ? 'vi-VN' : 'en-GB', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' })
}

/** Human label for a hyper-parameter value (numbers, lists, booleans, null). */
export function paramValue(v: unknown): string {
  const lang = activeLang()
  if (v == null) return translate(lang, 'common.none')
  // lists (layer widths, channels per stage) are written as lists, so [8, 16] never reads as a product
  if (Array.isArray(v)) return v.length ? `[${v.map(paramValue).join(', ')}]` : translate(lang, 'common.none')
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return int(v)
    return Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4 ? v.toExponential(0).replace('e-', 'e\u2212').replace('e+', 'e') : String(Number(v.toPrecision(3)))
  }
  if (typeof v === 'boolean') return translate(lang, v ? 'common.yes' : 'common.no')
  return String(v)
}

const PARAM_KEYS = [
  'learning_rate', 'l2', 'epochs', 'batch_size', 'hidden_sizes', 'momentum', 'p_drop', 'n_estimators', 'max_depth',
  'n_bins', 'max_features', 'var_smoothing', 'seed', 'lr_decay', 'tol', 'n_iter_no_change', 'early_stopping',
  'val_fraction', 'patience', 'min_delta', 'channels', 'kernel_size', 'pool', 'dense',
] as const

export function paramName(key: string): string {
  if ((PARAM_KEYS as readonly string[]).includes(key)) return translate(activeLang(), `params.${key as (typeof PARAM_KEYS)[number]}`)
  return key.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
}
