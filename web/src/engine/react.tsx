import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { CodedError, asCoded } from '../lib/errors'
import { fingerprint } from './bytes'
import type { BundleManifest, DatasetEntry, ModelKind, ReplayData } from './contract'
import type { Booted } from './boot'
import { matches, type Execution, type ModelRegistry, type ModelStatus } from './store'

export interface Bundle extends Booted {
  registry: ModelRegistry
}

export const BundleContext = createContext<Bundle | null>(null)

export function useBundle(): Bundle {
  const v = useContext(BundleContext)
  if (!v) throw new Error('Bundle is not available yet.')
  return v
}

export function useDataset(dataset: string): DatasetEntry {
  return useBundle().manifest.datasets[dataset]
}

/** Status of one model (idle / loading / ready / error); re-renders on change. */
export function useModelStatus(dataset: string, model: ModelKind): ModelStatus {
  const { registry } = useBundle()
  const all = useSyncExternalStore(registry.subscribe, registry.getSnapshot)
  return all[`${dataset}/${model}`] ?? { state: 'idle' }
}

export type ExecutionState =
  | { state: 'idle' }
  | { state: 'loading' }
  | { state: 'running' }
  | { state: 'ready'; execution: Execution }
  | { state: 'error'; error: CodedError }

/**
 * Executes `model` of `dataset` on `pixels` (784 values in [0, 1]; null = no input) and returns the
 * result ONLY while it belongs to exactly this model and this input. When the dataset, the model
 * or the pixels change the state drops to `loading` / `running` immediately, so no view can keep
 * showing the previous result. Models are loaded on demand when `autoLoad` is true.
 */
export function useExecution(dataset: string, model: ModelKind, pixels: Float32Array | null, autoLoad = true): ExecutionState {
  const { registry, manifest } = useBundle()
  const status = useModelStatus(dataset, model)
  const [result, setResult] = useState<{ exec: Execution } | { error: CodedError } | null>(null)
  const token = useRef(0)
  const inputId = pixels ? fingerprint(pixels) : null

  useEffect(() => {
    if (autoLoad && pixels) registry.load(dataset, model).catch(() => undefined)
  }, [registry, dataset, model, autoLoad, pixels])

  useEffect(() => {
    const my = ++token.current
    setResult(null)
    if (!pixels || status.state !== 'ready') return
    registry.execute(dataset, model, pixels).then(
      (exec) => token.current === my && setResult({ exec }),
      (e: unknown) => token.current === my && setResult({ error: asCoded(e) }),
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [registry, dataset, model, inputId, status.state])

  if (!pixels) return { state: 'idle' }
  if (status.state === 'error') return { state: 'error', error: new CodedError(status.key, status.params) }
  if (status.state !== 'ready') return { state: 'loading' }
  if (result && 'error' in result) return { state: 'error', error: result.error }
  if (result && matches(result.exec, manifest, dataset, model, pixels)) return { state: 'ready', execution: result.exec }
  return { state: 'running' }
}

/** Recorded training run of a model (genuine checkpoints and metrics), loaded when `enabled`. */
export function useReplay(dataset: string, model: ModelKind, enabled: boolean): { state: 'idle' | 'loading' | 'none' | 'error' | 'ready'; replay?: ReplayData; error?: CodedError } {
  const { registry } = useBundle()
  const [res, setRes] = useState<{ key: string; replay: ReplayData | null; error?: CodedError } | null>(null)
  const key = `${dataset}/${model}`
  useEffect(() => {
    if (!enabled) return
    let live = true
    registry.loadReplay(dataset, model).then(
      (replay) => live && setRes({ key, replay }),
      (e: unknown) => live && setRes({ key, replay: null, error: asCoded(e) }),
    )
    return () => {
      live = false
    }
  }, [registry, dataset, model, enabled, key])
  if (!enabled) return { state: 'idle' }
  if (!res || res.key !== key) return { state: 'loading' }
  if (res.error) return { state: 'error', error: res.error }
  return res.replay ? { state: 'ready', replay: res.replay } : { state: 'none' }
}

// --------------------------------------------------------------------------- playback
export function usePrefersReducedMotion(): boolean {
  const [v, setV] = useState(() => (typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) || false)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const on = () => setV(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return v
}

export const SPEEDS = [0.25, 0.5, 1, 2, 4] as const

export interface Playback {
  /** current step, 0 .. steps (steps = everything applied) */
  step: number
  steps: number
  playing: boolean
  speed: number
  /** true when the user asked for reduced motion: playback starts paused at the end and never autoplays */
  reduced: boolean
  play(): void
  pause(): void
  toggle(): void
  next(): void
  prev(): void
  seek(step: number): void
  /** go to step 0 and start playing (to the end) */
  replay(): void
  setSpeed(s: number): void
}

/**
 * Step-based playback clock for a computation of `steps` ordered operations. One step lasts
 * `baseMs / speed`. Without motion preference it plays once from the start whenever `resetKey`
 * changes (a new input or model); with `prefers-reduced-motion` it shows the final state and
 * only moves when the user steps or scrubs. Every state is reachable by stepping, so values can
 * be inspected without pausing at the right moment.
 */
export function usePlayback(steps: number, resetKey: string, baseMs = 40, autoplay = true): Playback {
  const reduced = usePrefersReducedMotion()
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const last = useRef(0)

  useEffect(() => {
    if (reduced || !autoplay) {
      setStep(steps)
      setPlaying(false)
    } else {
      setStep(0)
      setPlaying(steps > 0)
    }
  }, [resetKey, steps, reduced, autoplay])

  useEffect(() => {
    if (!playing) return
    let raf = 0
    last.current = performance.now()
    const tick = (now: number) => {
      const elapsed = now - last.current
      const per = baseMs / speed
      if (elapsed >= per) {
        const advance = Math.floor(elapsed / per)
        last.current += advance * per
        setStep((s) => {
          const n = Math.min(steps, s + advance)
          if (n >= steps) setPlaying(false)
          return n
        })
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, steps, baseMs])

  const play = useCallback(() => {
    setStep((s) => (s >= steps ? 0 : s))
    setPlaying(true)
  }, [steps])
  return useMemo<Playback>(
    () => ({
      step,
      steps,
      playing,
      speed,
      reduced,
      play,
      pause: () => setPlaying(false),
      toggle: () => (playing ? setPlaying(false) : play()),
      next: () => {
        setPlaying(false)
        setStep((s) => Math.min(steps, s + 1))
      },
      prev: () => {
        setPlaying(false)
        setStep((s) => Math.max(0, s - 1))
      },
      seek: (n: number) => {
        setPlaying(false)
        setStep(Math.max(0, Math.min(steps, Math.round(n))))
      },
      replay: () => {
        setStep(0)
        setPlaying(steps > 0)
      },
      setSpeed,
    }),
    [step, steps, playing, speed, reduced, play],
  )
}

export type { BundleManifest }
