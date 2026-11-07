import { useEffect, useRef } from 'react'
import { MNIST_SIZE } from '../lib/preprocess'
import { useI18n } from '../i18n'

interface Props {
  pixels: Float32Array | null
}

const N = MNIST_SIZE * MNIST_SIZE

function reducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * The exact 28 x 28 array that the models receive, drawn without smoothing.
 * Between updates every pixel eases toward its new value, so the digit visibly forms while you draw.
 */
export function Mnist28({ pixels }: Props) {
  const { t } = useI18n()
  const ref = useRef<HTMLCanvasElement>(null)
  const shown = useRef(new Float32Array(N))
  const raf = useRef(0)

  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    const target = pixels ?? new Float32Array(N)
    const img = ctx.createImageData(MNIST_SIZE, MNIST_SIZE)
    const cur = shown.current
    const draw = () => {
      for (let i = 0; i < N; i++) {
        const v = Math.round(cur[i] * 255)
        img.data[4 * i] = v
        img.data[4 * i + 1] = v
        img.data[4 * i + 2] = v
        img.data[4 * i + 3] = 255
      }
      ctx.putImageData(img, 0, 0)
    }
    cancelAnimationFrame(raf.current)
    if (reducedMotion()) {
      cur.set(target)
      draw()
      return
    }
    const step = () => {
      let far = 0
      for (let i = 0; i < N; i++) {
        const d = target[i] - cur[i]
        cur[i] += d * 0.3
        far = Math.max(far, Math.abs(d))
      }
      if (far < 0.01) cur.set(target)
      draw()
      if (far >= 0.01) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
  }, [pixels])

  return (
    <canvas
      ref={ref}
      className="mnist28"
      width={MNIST_SIZE}
      height={MNIST_SIZE}
      role="img"
      aria-label={pixels ? t('pad.previewAria') : t('pad.previewEmptyAria')}
    />
  )
}
