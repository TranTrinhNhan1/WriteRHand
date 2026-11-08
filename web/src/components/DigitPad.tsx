import { useCallback, useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react'
import { useI18n, type Key } from '../i18n'
import { grayFromRGBA, normalizeUpload, type BBox, type GrayImage } from '../lib/preprocess'
import { SAMPLE_STROKES, type SampleDigit } from '../lib/samples'

/** Side length of the offscreen capture used for analysis, independent of screen size and DPR. */
const CAPTURE = 280
const STROKE_WIDTH = 0.08
const MAX_UPLOAD_SIDE = 512
const MAX_UPLOAD_BYTES = 20 * 1024 * 1024

interface Pt {
  x: number
  y: number
}

export interface PadChange {
  /** White-on-black grayscale ready for `preprocessMnist`; null when the pad is blank. */
  image: GrayImage | null
  kind: 'draw' | 'upload'
  /** True while a stroke is still in progress. */
  drawing: boolean
  inverted?: boolean
  fileName?: string
}

export interface DigitPadHandle {
  clear(): void
  /** Draws a built-in digit stroke by stroke, as if by hand. */
  drawSample(digit: SampleDigit): void
  /**
   * Shows a ready-made 28 x 28 image (784 values in [0, 1]) on the pad, pixel for pixel. It does not
   * emit a change: the caller feeds those pixels to the models itself. Drawing, clearing or uploading
   * replaces it and emits as usual.
   */
  showImage(pixels: Float32Array): void
}

/** Crop rectangle in fractions of the drawing box (from the preprocessing bounding box). */
export interface CropRect {
  x: number
  y: number
  w: number
  h: number
}

export function cropOf(bbox: BBox, width: number, height: number): CropRect {
  return { x: bbox.x0 / width, y: bbox.y0 / height, w: (bbox.x1 - bbox.x0) / width, h: (bbox.y1 - bbox.y0) / height }
}

interface Props {
  onChange: (change: PadChange) => void
  /** Where the preprocessing found the ink; drawn as a thin frame so the 28 x 28 image makes sense. */
  crop?: CropRect | null
  /** Called on first pointer/keyboard contact; used to start loading the default model. */
  onIntent?: () => void
}

/** Paints strokes in `ink`; with a `paper` colour it fills the background first, otherwise it leaves it clear. */
function paintStrokes(ctx: CanvasRenderingContext2D, strokes: Pt[][], size: number, ink: string, paper: string | null) {
  if (paper) {
    ctx.fillStyle = paper
    ctx.fillRect(0, 0, size, size)
  } else {
    ctx.clearRect(0, 0, size, size)
  }
  ctx.strokeStyle = ink
  ctx.fillStyle = ink
  ctx.lineWidth = STROKE_WIDTH * size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  for (const s of strokes) {
    if (s.length === 0) continue
    if (s.length === 1) {
      ctx.beginPath()
      ctx.arc(s[0].x * size, s[0].y * size, (STROKE_WIDTH * size) / 2, 0, Math.PI * 2)
      ctx.fill()
      continue
    }
    ctx.beginPath()
    ctx.moveTo(s[0].x * size, s[0].y * size)
    for (let i = 1; i < s.length - 1; i++) {
      const mx = ((s[i].x + s[i + 1].x) / 2) * size
      const my = ((s[i].y + s[i + 1].y) / 2) * size
      ctx.quadraticCurveTo(s[i].x * size, s[i].y * size, mx, my)
    }
    const last = s[s.length - 1]
    ctx.lineTo(last.x * size, last.y * size)
    ctx.stroke()
  }
}

async function decodeImage(file: File): Promise<ImageBitmap | HTMLImageElement> {
  if ('createImageBitmap' in window) {
    try {
      return await createImageBitmap(file)
    } catch {
      /* fall through to <img> */
    }
  }
  const url = URL.createObjectURL(file)
  try {
    const img = new Image()
    img.src = url
    await img.decode()
    return img
  } finally {
    URL.revokeObjectURL(url)
  }
}

function sourceSize(src: ImageBitmap | HTMLImageElement): { w: number; h: number } {
  return src instanceof HTMLImageElement ? { w: src.naturalWidth, h: src.naturalHeight } : { w: src.width, h: src.height }
}

function reducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

export const DigitPad = forwardRef<DigitPadHandle, Props>(function DigitPad({ onChange, crop, onIntent }, handle) {
  const { t } = useI18n()
  const brushRef = useRef<HTMLDivElement>(null)
  const anim = useRef(0)
  const frameRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const strokes = useRef<Pt[][]>([])
  const active = useRef<{ id: number; stroke: Pt[] } | null>(null)
  const upload = useRef<{ canvas: HTMLCanvasElement; image: GrayImage; inverted: boolean; name: string } | null>(null)
  const shown = useRef<HTMLCanvasElement | null>(null)
  const raf = useRef(0)
  const timer = useRef(0)
  const lastEmit = useRef(0)
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange

  const [hasContent, setHasContent] = useState(false)
  const [canUndo, setCanUndo] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<Key | null>(null)
  const [note, setNote] = useState<Key | null>(null)

  const paint = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const css = canvas.clientWidth
    const dpr = window.devicePixelRatio || 1
    const px = Math.max(1, Math.round(css * dpr))
    if (canvas.width !== px || canvas.height !== px) {
      canvas.width = px
      canvas.height = px
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    const ink = getComputedStyle(canvas).getPropertyValue('--pad-ink').trim() || '#1b1f1a'
    if (shown.current) {
      // A ready-made 28 x 28 image: scaled up without smoothing so every pixel stays visible.
      ctx.clearRect(0, 0, px, px)
      const tint = document.createElement('canvas')
      tint.width = px
      tint.height = px
      const tctx = tint.getContext('2d')!
      tctx.imageSmoothingEnabled = false
      tctx.drawImage(shown.current, 0, 0, px, px)
      tctx.globalCompositeOperation = 'source-in'
      tctx.fillStyle = ink
      tctx.fillRect(0, 0, px, px)
      ctx.drawImage(tint, 0, 0)
    } else if (upload.current) {
      // The uploaded picture is stored as an ink mask (white with alpha). Tint it with the ink colour.
      ctx.clearRect(0, 0, px, px)
      const { canvas: uc } = upload.current
      const k = Math.min(px / uc.width, px / uc.height)
      const w = uc.width * k
      const h = uc.height * k
      const tint = document.createElement('canvas')
      tint.width = px
      tint.height = px
      const tctx = tint.getContext('2d')!
      tctx.imageSmoothingQuality = 'high'
      tctx.drawImage(uc, (px - w) / 2, (px - h) / 2, w, h)
      tctx.globalCompositeOperation = 'source-in'
      tctx.fillStyle = ink
      tctx.fillRect(0, 0, px, px)
      ctx.drawImage(tint, 0, 0)
    } else {
      paintStrokes(ctx, strokes.current, px, ink, null)
    }
  }, [])

  const schedulePaint = useCallback(() => {
    if (raf.current) return
    raf.current = requestAnimationFrame(() => {
      raf.current = 0
      paint()
    })
  }, [paint])

  const capture = useCallback((): PadChange => {
    if (upload.current) {
      return {
        image: upload.current.image,
        kind: 'upload',
        drawing: false,
        inverted: upload.current.inverted,
        fileName: upload.current.name,
      }
    }
    const done = strokes.current.length > 0
    if (!done) return { image: null, kind: 'draw', drawing: false }
    const off = document.createElement('canvas')
    off.width = CAPTURE
    off.height = CAPTURE
    const ctx = off.getContext('2d', { willReadFrequently: true })
    if (!ctx) return { image: null, kind: 'draw', drawing: false }
    paintStrokes(ctx, strokes.current, CAPTURE, '#fff', '#000')
    const data = ctx.getImageData(0, 0, CAPTURE, CAPTURE)
    return { image: grayFromRGBA(data.data, CAPTURE, CAPTURE), kind: 'draw', drawing: active.current !== null }
  }, [])

  const emit = useCallback(() => {
    window.clearTimeout(timer.current)
    timer.current = 0
    lastEmit.current = performance.now()
    onChangeRef.current(capture())
  }, [capture])

  /** Emits at most every ~110 ms while drawing (trailing), immediately otherwise. */
  const emitSoon = useCallback(() => {
    if (timer.current) return
    const wait = Math.max(0, 110 - (performance.now() - lastEmit.current))
    timer.current = window.setTimeout(emit, wait)
  }, [emit])

  const stopAnim = useCallback(() => {
    cancelAnimationFrame(anim.current)
    anim.current = 0
  }, [])

  const clear = useCallback(() => {
    stopAnim()
    strokes.current = []
    active.current = null
    upload.current = null
    shown.current = null
    setHasContent(false)
    setCanUndo(false)
    setError(null)
    setNote(null)
    paint()
    emit()
  }, [paint, emit, stopAnim])

  const drawSample = useCallback(
    (digit: SampleDigit) => {
      onIntent?.()
      stopAnim()
      active.current = null
      upload.current = null
      shown.current = null
      setError(null)
      setNote(null)
      const template = SAMPLE_STROKES[digit]
      const total = template.reduce((n, st) => n + st.length, 0)
      setHasContent(true)
      setCanUndo(true)
      const upTo = (count: number) => {
        let left = count
        strokes.current = template
          .map((st) => {
            const take = Math.max(0, Math.min(st.length, left))
            left -= take
            return st.slice(0, take)
          })
          .filter((st) => st.length > 0)
      }
      if (reducedMotion()) {
        upTo(total)
        paint()
        emit()
        return
      }
      const t0 = performance.now()
      const duration = 260 + total * 7
      const tick = (now: number) => {
        const k = Math.min(1, (now - t0) / duration)
        upTo(Math.max(1, Math.round(k * total)))
        schedulePaint()
        if (k < 1) {
          emitSoon()
          anim.current = requestAnimationFrame(tick)
        } else {
          anim.current = 0
          emit()
        }
      }
      strokes.current = []
      anim.current = requestAnimationFrame(tick)
    },
    [emit, emitSoon, onIntent, paint, schedulePaint, stopAnim],
  )

  const showImage = useCallback(
    (pixels: Float32Array) => {
      stopAnim()
      active.current = null
      upload.current = null
      strokes.current = []
      const c = document.createElement('canvas')
      c.width = 28
      c.height = 28
      const cctx = c.getContext('2d')!
      const img = cctx.createImageData(28, 28)
      for (let i = 0; i < 784; i++) {
        img.data[4 * i] = 255
        img.data[4 * i + 1] = 255
        img.data[4 * i + 2] = 255
        img.data[4 * i + 3] = Math.round(Math.min(1, Math.max(0, pixels[i])) * 255)
      }
      cctx.putImageData(img, 0, 0)
      shown.current = c
      setHasContent(true)
      setCanUndo(false)
      setError(null)
      setNote(null)
      window.clearTimeout(timer.current)
      timer.current = 0
      paint()
    },
    [paint, stopAnim],
  )

  useImperativeHandle(handle, () => ({ clear, drawSample, showImage }), [clear, drawSample, showImage])

  useEffect(() => {
    paint()
    const frame = frameRef.current
    if (!frame) return
    const ro = new ResizeObserver(schedulePaint)
    ro.observe(frame)
    const mq = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`)
    mq.addEventListener('change', schedulePaint)
    const dark = window.matchMedia('(prefers-color-scheme: dark)')
    dark.addEventListener('change', schedulePaint)
    return () => {
      ro.disconnect()
      mq.removeEventListener('change', schedulePaint)
      dark.removeEventListener('change', schedulePaint)
      cancelAnimationFrame(anim.current)
      cancelAnimationFrame(raf.current)
      window.clearTimeout(timer.current)
    }
  }, [paint, schedulePaint])

  const toPoint = (e: { clientX: number; clientY: number }): Pt => {
    const r = canvasRef.current!.getBoundingClientRect()
    return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) }
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return
    if (active.current) return
    onIntent?.()
    if (anim.current) {
      // A sample is still being drawn: keep what is there and let the pen take over.
      stopAnim()
    }
    if (upload.current || shown.current) {
      upload.current = null
      shown.current = null
      strokes.current = []
    }
    e.currentTarget.setPointerCapture(e.pointerId)
    const stroke = [toPoint(e)]
    strokes.current.push(stroke)
    active.current = { id: e.pointerId, stroke }
    setHasContent(true)
    setCanUndo(true)
    setError(null)
    setNote(null)
    schedulePaint()
    emitSoon()
  }

  const moveBrush = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const b = brushRef.current
    if (!b) return
    if (e.pointerType === 'touch') {
      b.style.opacity = '0'
      return
    }
    const r = e.currentTarget.getBoundingClientRect()
    b.style.transform = `translate(${e.clientX - r.left}px, ${e.clientY - r.top}px) translate(-50%, -50%)`
    b.style.opacity = '1'
  }

  const hideBrush = () => {
    if (brushRef.current) brushRef.current.style.opacity = '0'
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    moveBrush(e)
    const a = active.current
    if (!a || a.id !== e.pointerId) return
    const native = e.nativeEvent
    const events = typeof native.getCoalescedEvents === 'function' ? native.getCoalescedEvents() : []
    const list = events.length ? events : [native]
    for (const ev of list) {
      const p = toPoint(ev)
      const prev = a.stroke[a.stroke.length - 1]
      if (Math.hypot(p.x - prev.x, p.y - prev.y) >= 0.003) a.stroke.push(p)
    }
    schedulePaint()
    emitSoon()
  }

  const endStroke = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const a = active.current
    if (!a || a.id !== e.pointerId) return
    active.current = null
    schedulePaint()
    emit()
  }

  const undo = () => {
    if (upload.current || strokes.current.length === 0) return
    strokes.current.pop()
    setHasContent(strokes.current.length > 0)
    setCanUndo(strokes.current.length > 0)
    paint()
    emit()
  }

  const loadFile = async (file: File | undefined | null) => {
    if (!file) return
    onIntent?.()
    setError(null)
    if (!file.type.startsWith('image/')) {
      setError('pad.notImage')
      return
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError('pad.tooBig')
      return
    }
    stopAnim()
    try {
      const src = await decodeImage(file)
      const { w, h } = sourceSize(src)
      const k = Math.min(1, MAX_UPLOAD_SIDE / Math.max(w, h))
      const cw = Math.max(1, Math.round(w * k))
      const ch = Math.max(1, Math.round(h * k))
      const tmp = document.createElement('canvas')
      tmp.width = cw
      tmp.height = ch
      const tctx = tmp.getContext('2d', { willReadFrequently: true })
      if (!tctx) throw new Error('no 2d context')
      tctx.fillStyle = '#fff'
      tctx.fillRect(0, 0, cw, ch)
      tctx.imageSmoothingQuality = 'high'
      tctx.drawImage(src, 0, 0, cw, ch)
      if ('close' in src) src.close()
      const gray = grayFromRGBA(tctx.getImageData(0, 0, cw, ch).data, cw, ch)
      const { image, inverted } = normalizeUpload(gray)
      // Keep an ink mask for display (white, with the ink strength as alpha), so the pad shows what the pipeline will see.
      const view = document.createElement('canvas')
      view.width = cw
      view.height = ch
      const vctx = view.getContext('2d')!
      const out = vctx.createImageData(cw, ch)
      for (let i = 0; i < image.data.length; i++) {
        out.data[4 * i] = 255
        out.data[4 * i + 1] = 255
        out.data[4 * i + 2] = 255
        out.data[4 * i + 3] = Math.round(image.data[i] * 255)
      }
      vctx.putImageData(out, 0, 0)
      strokes.current = []
      shown.current = null
      upload.current = { canvas: view, image, inverted, name: file.name }
      setHasContent(true)
      setCanUndo(false)
      setNote(inverted ? 'pad.inverted' : null)
      paint()
      emit()
    } catch {
      setError('pad.unreadable')
    }
  }

  return (
    <div className="pad">
      <div
        ref={frameRef}
        className={`pad__frame${dragging ? ' is-dragging' : ''}${hasContent ? ' has-ink' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          void loadFile(e.dataTransfer.files[0])
        }}
      >
        <i className="pad__mark pad__mark--tl" aria-hidden="true" />
        <i className="pad__mark pad__mark--tr" aria-hidden="true" />
        <i className="pad__mark pad__mark--bl" aria-hidden="true" />
        <i className="pad__mark pad__mark--br" aria-hidden="true" />
        <canvas
          ref={canvasRef}
          className="pad__canvas"
          role="img"
          aria-label={hasContent ? t('pad.filledAria') : t('pad.emptyAria')}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endStroke}
          onPointerCancel={endStroke}
          onPointerEnter={() => onIntent?.()}
          onPointerLeave={hideBrush}
          onContextMenu={(e) => e.preventDefault()}
        />
        {crop && !upload.current && !shown.current && (
          <div
            className="pad__crop"
            aria-hidden="true"
            title={t('pad.crop')}
            style={{ left: `${crop.x * 100}%`, top: `${crop.y * 100}%`, width: `${crop.w * 100}%`, height: `${crop.h * 100}%` }}
          />
        )}
        <div ref={brushRef} className="pad__brush" aria-hidden="true" style={{ width: `${STROKE_WIDTH * 100}%`, aspectRatio: '1' }} />
        {!hasContent && (
          <p className="pad__hint" aria-hidden="true">
            {t('pad.hint')}
          </p>
        )}
        {dragging && <p className="pad__drop">{t('pad.drop')}</p>}
      </div>
      <div className="pad__tools">
        <button type="button" className="btn btn--quiet" onClick={clear} disabled={!hasContent}>
          {t('pad.clear')}
        </button>
        <button type="button" className="btn btn--quiet" onClick={undo} disabled={!canUndo}>
          {t('pad.undo')}
        </button>
        <button type="button" className="btn btn--quiet" onClick={() => fileRef.current?.click()}>
          {t('pad.upload')}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="visually-hidden"
          aria-label={t('pad.uploadAria')}
          tabIndex={-1}
          onChange={(e) => {
            void loadFile(e.target.files?.[0])
            e.target.value = ''
          }}
        />
      </div>
      <div className="pad__messages" aria-live="polite">
        {error && (
          <p role="alert" className="msg msg--error">
            {t(error)}
          </p>
        )}
        {!error && note && <p className="msg">{t(note)}</p>}
      </div>
    </div>
  )
})
