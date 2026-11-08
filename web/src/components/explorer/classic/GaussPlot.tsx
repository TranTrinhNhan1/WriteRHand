import { useI18n } from '../../../i18n'
import { curveDomain, fmt, gaussianPdf, sampleCurve } from './model'

interface Props {
  /** mean and variance of this pixel for every class (standardised units) */
  mus: ArrayLike<number>
  vars: ArrayLike<number>
  /** standardised input value at the pixel */
  x: number
  /** class drawn in the accent colour */
  k: number
  className: string
}

const W = 320
const H = 132
const PAD = { l: 8, r: 8, t: 10, b: 22 }
const SAMPLES = 120

/**
 * The class-conditional Gaussians of one pixel, computed from the model's real means and variances:
 * the selected class in the accent colour, the other classes as thin lines, the input value marked.
 */
export function GaussPlot({ mus, vars, x, k, className }: Props) {
  const { t } = useI18n()
  const [lo, hi] = curveDomain(mus, vars, x)
  let peak = 0
  for (let i = 0; i < mus.length; i++) peak = Math.max(peak, 1 / Math.sqrt(2 * Math.PI * vars[i]))
  const X = (v: number) => PAD.l + ((v - lo) / (hi - lo)) * (W - PAD.l - PAD.r)
  const Y = (d: number) => H - PAD.b - (d / peak) * (H - PAD.t - PAD.b)
  const line = (mu: number, v: number) => {
    const ys = sampleCurve(mu, v, lo, hi, SAMPLES)
    return Array.from(ys, (d, i) => `${i ? 'L' : 'M'}${X(lo + ((hi - lo) * i) / (SAMPLES - 1)).toFixed(1)},${Y(d).toFixed(1)}`).join('')
  }
  const dx = gaussianPdf(x, mus[k], vars[k])
  const others = Array.from({ length: mus.length }, (_, c) => c).filter((c) => c !== k)
  const xx = X(x)
  return (
    <figure className="exc-gauss">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t('ex.nb.gaussAria', { name: className, x: fmt(x, 3), mu: fmt(mus[k], 3), v: fmt(vars[k], 3), d: fmt(dx, 4) })}>
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} className="exc-gauss__axis" />
        {others.map((c) => (
          <path key={c} d={line(mus[c], vars[c])} className="exc-gauss__other" />
        ))}
        <path d={`${line(mus[k], vars[k])}L${X(hi)},${Y(0)}L${X(lo)},${Y(0)}Z`} className="exc-gauss__area" />
        <path d={line(mus[k], vars[k])} className="exc-gauss__sel" />
        <line x1={X(mus[k])} x2={X(mus[k])} y1={Y(0)} y2={Y(1 / Math.sqrt(2 * Math.PI * vars[k]))} className="exc-gauss__mu" />
        <line x1={xx} x2={xx} y1={PAD.t - 4} y2={H - PAD.b} className="exc-gauss__x" />
        <circle cx={xx} cy={Y(dx)} r={3.5} className="exc-gauss__dot" />
        <text x={X(mus[k])} y={H - 6} textAnchor="middle" className="exc-gauss__lbl">
          μ
        </text>
        <text x={xx > W - 30 ? xx - 5 : xx + 5} y={PAD.t + 4} textAnchor={xx > W - 30 ? 'end' : 'start'} className="exc-gauss__lbl exc-gauss__lbl--x">
          x
        </text>
      </svg>
      <figcaption className="exc-fig__cap">{t('ex.nb.gaussCap', { name: className })}</figcaption>
    </figure>
  )
}
