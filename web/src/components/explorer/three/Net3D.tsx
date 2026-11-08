import { ArrowCounterClockwise, Minus, Plus } from '@phosphor-icons/react'
import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { useI18n } from '../../../i18n'
import { usePrefersReducedMotion } from '../../../engine/react'
import type { Colors3D, Net3DHandle, Pick3D } from './engine3d'
import type { NetSpec } from './spec'

type Engine = typeof import('./engine3d')
let engine: Promise<Engine> | null = null
const loadEngine = () => (engine ??= import('./engine3d'))

/**
 * The 3D view sits on the dark instrument screen in both themes, like every heat map of the explorer
 * (HeatMap.tsx): the same fixed screen, lime / orange and ink colours, so a value looks the same everywhere.
 */
function readColors(el: Element): Colors3D {
  const cs = getComputedStyle(el)
  const v = (n: string) => cs.getPropertyValue(n).trim()
  return {
    bg: v('--pad-paper') || '#070908',
    ink: v('--pad-ink') || '#f1f5ec',
    ink2: '#c9d1ca',
    ink3: '#8a958d',
    line: '#3a4540',
    sunken: '#1f2622',
    surface: '#111513',
    pos: '#c6f135',
    neg: '#ff7a59',
    paper: '#070908',
    padInk: '#f1f5ec',
    mono: v('--mono'),
  }
}

interface Props {
  spec: NetSpec
  /** trace layers reached by the playback (null: everything) */
  revealed: ReadonlySet<string> | null
  selected: Pick3D | null
  onPick: (p: Pick3D) => void
  /** accessible summary of what is drawn */
  label: string
  /** WebGL missing or failed: the caller shows the 2D view instead */
  onUnavailable: () => void
}

const STEP = Math.PI / 24

/**
 * WebGL canvas of a layered network. Rebuilt when the data or the theme changes, disposed on unmount;
 * renders only on demand and only while visible. Keyboard: arrows rotate, + / − zoom, 0 resets.
 */
export default function Net3D({ spec, revealed, selected, onPick, label, onUnavailable }: Props) {
  const { t } = useI18n()
  const host = useRef<HTMLDivElement>(null)
  const handle = useRef<Net3DHandle | null>(null)
  const reduced = usePrefersReducedMotion()
  const [theme, setTheme] = useState(0)
  const [ready, setReady] = useState(false)
  const latest = useRef({ revealed, selected })
  latest.current = { revealed, selected }
  const down = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const mo = new MutationObserver(() => setTheme((n) => n + 1))
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => mo.disconnect()
  }, [])

  // build (and tear down) the scene
  useEffect(() => {
    const el = host.current
    if (!el) return
    let cancelled = false
    let ro: ResizeObserver | null = null
    let io: IntersectionObserver | null = null
    let onVis: (() => void) | null = null
    loadEngine()
      .then((E) => {
        if (cancelled) return
        if (!E.webglAvailable()) {
          onUnavailable()
          return
        }
        let h: Net3DHandle
        try {
          h = E.createNet3D(el, spec, readColors(el), { reducedMotion: reduced })
        } catch {
          onUnavailable()
          return
        }
        handle.current = h
        h.setRevealed(latest.current.revealed)
        h.setSelected(latest.current.selected)
        // the stage is as tall as the network needs at this width (no tall, mostly empty box)
        let lastW = -1
        const fitHeight = (w: number) => {
          if (Math.abs(w - lastW) < 1) return
          lastW = w
          const min = w < 520 ? 220 : 280
          el.style.height = `${Math.round(Math.max(min, Math.min(520, h.preferredHeight(w))))}px`
        }
        const r = el.getBoundingClientRect()
        fitHeight(r.width)
        const r2 = el.getBoundingClientRect()
        h.resize(r2.width, r2.height)
        ro = new ResizeObserver(([e]) => {
          fitHeight(e.contentRect.width)
          h.resize(e.contentRect.width, e.contentRect.height)
        })
        ro.observe(el)
        let onScreen = true
        const apply = () => h.setActive(onScreen && document.visibilityState === 'visible')
        io = new IntersectionObserver(([e]) => {
          onScreen = e.isIntersecting
          apply()
        })
        io.observe(el)
        onVis = apply
        document.addEventListener('visibilitychange', onVis)
        setReady(true)
      })
      .catch(() => !cancelled && onUnavailable())
    return () => {
      cancelled = true
      ro?.disconnect()
      io?.disconnect()
      if (onVis) document.removeEventListener('visibilitychange', onVis)
      handle.current?.dispose()
      handle.current = null
      el.style.height = ''
      setReady(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, theme, reduced])

  useEffect(() => handle.current?.setRevealed(revealed), [revealed, ready])
  useEffect(() => handle.current?.setSelected(selected), [selected, ready])

  const onKey = (e: KeyboardEvent) => {
    const h = handle.current
    if (!h) return
    const k = e.key
    if (k === 'ArrowLeft') h.rotate(-STEP, 0)
    else if (k === 'ArrowRight') h.rotate(STEP, 0)
    else if (k === 'ArrowUp') h.rotate(0, -STEP)
    else if (k === 'ArrowDown') h.rotate(0, STEP)
    else if (k === '+' || k === '=') h.zoom(0.85)
    else if (k === '-' || k === '_') h.zoom(1 / 0.85)
    else if (k === '0' || k === 'Home') h.reset()
    else return
    e.preventDefault()
  }
  const onDown = (e: PointerEvent) => (down.current = { x: e.clientX, y: e.clientY })
  const onUp = (e: PointerEvent) => {
    const d = down.current
    down.current = null
    if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return
    const p = handle.current?.pick(e.clientX, e.clientY)
    if (p) onPick(p)
  }

  return (
    <div className="ex-3d">
      <div
        ref={host}
        className="ex-3d__stage"
        tabIndex={0}
        role="application"
        aria-roledescription={t('ex.three.role')}
        aria-label={label}
        aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown + - 0"
        aria-busy={!ready}
        onKeyDown={onKey}
        onPointerDown={onDown}
        onPointerUp={onUp}
      />
      <div className="ex-3d__tools" role="group" aria-label={t('ex.three.view')}>
        <button type="button" className="ex-3d__btn" onClick={() => handle.current?.zoom(0.85)} aria-label={t('ex.three.zoomIn')} title={t('ex.three.zoomIn')}>
          <Plus size={14} aria-hidden="true" />
        </button>
        <button type="button" className="ex-3d__btn" onClick={() => handle.current?.zoom(1 / 0.85)} aria-label={t('ex.three.zoomOut')} title={t('ex.three.zoomOut')}>
          <Minus size={14} aria-hidden="true" />
        </button>
        <button type="button" className="ex-3d__btn" onClick={() => handle.current?.reset()} aria-label={t('ex.three.reset')} title={t('ex.three.reset')}>
          <ArrowCounterClockwise size={14} aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}
