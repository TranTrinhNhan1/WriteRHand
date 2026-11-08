import { useEffect, useRef } from 'react'

/** A 28 x 28 image drawn pixel for pixel (white ink on black, like the models see it). */
export function Thumb28({ pixels, className }: { pixels: Float32Array; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(28, 28)
    for (let i = 0; i < 784; i++) {
      const v = Math.round(Math.min(1, Math.max(0, pixels[i])) * 255)
      img.data[4 * i] = v
      img.data[4 * i + 1] = v
      img.data[4 * i + 2] = v
      img.data[4 * i + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [pixels])
  return <canvas ref={ref} width={28} height={28} className={className ?? 'thumb28'} aria-hidden="true" />
}
