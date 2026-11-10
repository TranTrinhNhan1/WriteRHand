import { useEffect, useId, useMemo, useRef } from 'react'
import { ChartFigure } from '../components/charts/ChartFigure'
import { useCssColors } from '../components/explorer/classic/useCssColors'
import { Segmented } from '../components/Segmented'
import { useBundle } from '../engine/react'
import { useI18n, type Key } from '../i18n'
import { useChoice } from '../lib/choice'
import { IMAGE_PIXELS, IMAGE_SIZE, decodeImage, type DatasetIntro } from '../lib/datasets'
import { int, pct } from '../lib/format'
import { useElementWidth } from '../lib/hooks'
import { linearTicks } from '../lib/scale'
import { datasetIdsOf } from '../lib/selection'

const DATASET_TEXT: Record<string, Key> = {
  mnist: 'about.data.mnist',
  fashion_mnist: 'about.data.fashion_mnist',
  emnist_digits: 'about.data.emnist_digits',
}

/** grey level 0 is drawn in the paper colour, 255 in the ink colour (the page's own tokens, so both themes work) */
const COLOR_TOKENS = ['--ink', '--sunken'] as const
type Rgb = [number, number, number]

function rgb(css: string): Rgb {
  const h = css.trim().replace(/^#/, '')
  if (/^[0-9a-f]{6}$/i.test(h)) return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as Rgb
  if (/^[0-9a-f]{3}$/i.test(h)) return [0, 1, 2].map((i) => parseInt(h[i] + h[i], 16)) as Rgb
  const m = css.match(/\d+(\.\d+)?/g)
  return m && m.length >= 3 ? (m.slice(0, 3).map(Number) as Rgb) : [128, 128, 128]
}

/** One 28 x 28 image, drawn pixel for pixel (the CSS scales it up without smoothing). Decoded only when drawn. */
function GreyImage({ b64, ink, paper, label, className }: { b64: string; ink: Rgb; paper: Rgb; label: string; className: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const pixels = useMemo(() => decodeImage(b64), [b64])
  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE)
    for (let i = 0; i < IMAGE_PIXELS; i++) {
      const a = pixels[i] / 255
      for (let c = 0; c < 3; c++) img.data[4 * i + c] = Math.round(paper[c] + (ink[c] - paper[c]) * a)
      img.data[4 * i + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [pixels, ink, paper])
  return <canvas ref={ref} width={IMAGE_SIZE} height={IMAGE_SIZE} className={className} role="img" aria-label={label} />
}

function ClassGallery({ data }: { data: DatasetIntro }) {
  const { t } = useI18n()
  const colors = useCssColors(COLOR_TOKENS)
  const ink = useMemo(() => rgb(colors['--ink']), [colors])
  const paper = useMemo(() => rgb(colors['--sunken']), [colors])
  const titleId = useId()
  return (
    <section className="dsgal" aria-labelledby={titleId}>
      <div className="chart__head">
        <h4 id={titleId} className="chart__title">
          {t('about.data.gallery')}
        </h4>
        <p className="caption">{t('about.data.galleryCaption')}</p>
      </div>
      <div className="dsgal__list">
        {data.classes.map((c) => (
          <figure key={c.label} className="dsclass">
            <figcaption className="dsclass__cap">
              <span className="dsclass__label">{c.label}</span>
              <span className="dsclass__n num">{t('about.data.trainCount', { n: int(c.n.train) })}</span>
            </figcaption>
            <div className="dsclass__mean">
              <GreyImage
                b64={c.mean}
                ink={ink}
                paper={paper}
                className="dsimg dsimg--mean"
                label={t('about.data.meanAria', { title: data.title, label: c.label })}
              />
              <span className="dsclass__tag" aria-hidden="true">
                {t('about.data.mean')}
              </span>
            </div>
            <div className="dsclass__samples">
              {c.samples.map((s, i) => (
                <GreyImage
                  key={i}
                  b64={s}
                  ink={ink}
                  paper={paper}
                  className="dsimg"
                  label={t('about.data.sampleAria', { title: data.title, label: c.label, i: i + 1 })}
                />
              ))}
            </div>
          </figure>
        ))}
      </div>
    </section>
  )
}

/** Horizontal bars of the training images per class, on an axis from 0 so that any imbalance is shown at its true size. */
function ClassBalance({ data }: { data: DatasetIntro }) {
  const { t } = useI18n()
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const titleId = useId()
  const counts = data.classes.map((c) => c.n.train)
  const ticks = linearTicks(0, Math.max(...counts), width < 420 ? 3 : 5)
  const top = ticks[ticks.length - 1]
  const longest = Math.max(...data.classes.map((c) => c.label.length))
  const LABEL = Math.ceil(longest * 7.4) + 14
  const VALUE = 48
  const ROW = 22
  const BAR = 14
  const AXIS = 38
  const plotW = Math.max(80, width - LABEL - VALUE)
  const sx = (v: number) => LABEL + (v / top) * plotW
  const bottom = data.classes.length * ROW
  return (
    <div ref={ref} className="chart__svg dsbal">
      {width > 0 && (
        <svg width={width} height={bottom + AXIS} role="img" aria-labelledby={titleId}>
          <title id={titleId}>{t('about.data.balanceAria', { title: data.title })}</title>
          {ticks.map((v) => (
            <g key={v}>
              <line x1={sx(v)} x2={sx(v)} y1={0} y2={bottom} className="grid" />
              <text x={sx(v)} y={bottom + 14} textAnchor="middle" className="axis-text">
                {int(v)}
              </text>
            </g>
          ))}
          <text x={LABEL + plotW / 2} y={bottom + AXIS - 4} textAnchor="middle" className="axis-title">
            {t('about.data.balanceAxis')}
          </text>
          {data.classes.map((c, i) => {
            const y = i * ROW + (ROW - BAR) / 2
            return (
              <g key={c.label}>
                <text x={LABEL - 8} y={y + BAR - 3} textAnchor="end" className="axis-text dsbal__name">
                  {c.label}
                </text>
                <rect x={sx(0)} y={y} width={Math.max(1, sx(c.n.train) - sx(0))} height={BAR} rx={2} className="dsbal__bar" />
                <text x={sx(c.n.train) + 6} y={y + BAR - 3} className="value-text">
                  {int(c.n.train)}
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

/** The About page's introduction of the datasets: facts, every class's mean and sample images, and the class balance. */
export function DatasetsSection() {
  const { t } = useI18n()
  const { manifest, datasets } = useBundle()
  const { dataset, setDataset } = useChoice()
  const ids = datasetIdsOf(manifest)
  const ds = manifest.datasets[dataset]
  const d = ds.dataset
  const intro = datasets[dataset]
  const n = { train: d?.n_train ?? intro?.n.train, val: d?.n_val ?? intro?.n.val, test: d?.n_test ?? intro?.n.test }
  const facts: { label: string; value: string; sub?: string }[] = [
    { label: t('about.data.size'), value: `${IMAGE_SIZE} × ${IMAGE_SIZE}`, sub: t('about.data.sizeSub') },
    { label: t('about.data.classCount'), value: String(ds.classes.length) },
    { label: t('about.split.train'), value: int(n.train) },
    { label: t('about.split.val'), value: int(n.val) },
    { label: t('about.split.test'), value: int(n.test) },
    ...(intro ? [{ label: t('about.data.ink'), value: pct(intro.ink_fraction, 1), sub: t('about.data.inkSub') }] : []),
  ]
  return (
    <section aria-labelledby="a-data" className="dsintro">
      <h2 id="a-data" className="h2">
        {t('about.data.title')}
      </h2>
      <p>{t('about.data.text')}</p>
      {ids.length > 1 && (
        <Segmented legend={t('about.data.legend')} value={dataset} onChange={setDataset} options={ids.map((id) => ({ value: id, label: manifest.datasets[id].title }))} />
      )}
      <div className="dsabout" data-dataset={dataset}>
        <h3 className="h3">{ds.title}</h3>
        {DATASET_TEXT[dataset] && <p>{t(DATASET_TEXT[dataset])}</p>}
        <dl className="dsfacts" aria-label={t('about.data.facts', { title: ds.title })}>
          {facts.map((f) => (
            <div key={f.label}>
              <dt>{f.label}</dt>
              <dd>
                <span className="dsfacts__v num">{f.value}</span>
                {f.sub && <span className="dsfacts__sub">{f.sub}</span>}
              </dd>
            </div>
          ))}
        </dl>
        <dl className="kv">
          <div>
            <dt>{t('about.data.classes')}</dt>
            <dd>{ds.classes.join(', ')}</dd>
          </div>
          <div>
            <dt>{t('about.data.source')}</dt>
            <dd>{ds.source}</dd>
          </div>
        </dl>
        {intro && <ClassGallery key={dataset} data={intro} />}
        {intro && (
          <ChartFigure
            level={4}
            title={t('about.data.balance')}
            table={{
              columns: [t('about.data.classCol'), t('about.split.train'), t('about.split.val'), t('about.split.test')],
              rows: intro.classes.map((c) => [c.label, int(c.n.train), int(c.n.val), int(c.n.test)]),
            }}
          >
            <ClassBalance data={intro} />
          </ChartFigure>
        )}
      </div>
    </section>
  )
}
