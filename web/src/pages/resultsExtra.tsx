/**
 * Optional sections of the Results page. Each one is rendered only when its data exists in the
 * dataset's results file (see `sanitizeResults`); nothing here invents numbers or causes.
 */
import { useId, useState, type CSSProperties } from 'react'
import { Section } from '../components/Section'
import { Segmented } from '../components/Segmented'
import { ChartFigure } from '../components/charts/ChartFigure'
import { LineChart } from '../components/charts/LineChart'
import type { PairSide, Pairs } from '../lib/extraModels'
import { MODEL_ORDER, isModelKind } from '../engine/contract'
import { useBundle } from '../engine/react'
import { useI18n, type Key } from '../i18n'
import { bytes, int, ms, num, pct, seconds } from '../lib/format'
import { useElementWidth } from '../lib/hooks'
import { augRows, augStressKeys, datasetMatrix, datasetSpreads, type AugRow, type AugVerdict } from '../lib/results'
import { datasetIdsOf } from '../lib/selection'
import { finite } from '../lib/scale'
import type { AugStrengths, AugStudy, CnnVsMlp, CnnVsMlpRow, CrossEval, Results } from '../lib/types'

type T = ReturnType<typeof useI18n>['t']

const TRANSFORM_KEYS = ['rotation', 'translation', 'scaling', 'gaussian_noise', 'salt_pepper', 'geometric', 'all', 'baseline'] as const

/** Name of an augmentation condition or perturbation; unknown names are shown as written. */
export function condName(t: T, name: string): string {
  return (TRANSFORM_KEYS as readonly string[]).includes(name) ? t(`results.aug.cond.${name as (typeof TRANSFORM_KEYS)[number]}`) : name.replace(/_/g, ' ')
}

/** Strength of a transformation in its own unit. */
export function strength(name: string, v: AugStrengths[string] | undefined): string {
  if (v == null) return '–'
  if (Array.isArray(v)) return v.map((x) => String(x)).join('–')
  switch (name) {
    case 'rotation':
      return `±${v}°`
    case 'translation':
      return `±${v} px`
    case 'gaussian_noise':
      return `σ = ${v}`
    case 'salt_pepper':
      return pct(v, v * 100 < 1 ? 1 : 0)
    default:
      return String(v)
  }
}

const pts = (v: number | null | undefined, d = 2) => (finite(v) ? `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(d)}` : '–')

const VERDICT_KEY: Record<AugVerdict, Key> = {
  baseline: 'results.aug.verdict.baseline',
  helps: 'results.aug.verdict.helps',
  unclear: 'results.aug.verdict.unclear',
  hurts: 'results.aug.verdict.hurts',
}

// --------------------------------------------------------------------------- augmentation
function AugDots({ rows, label }: { rows: AugRow[]; label: string }) {
  const { t } = useI18n()
  const [ref, width] = useElementWidth<HTMLDivElement>()
  const titleId = useId()
  const base = rows.find((r) => r.name === 'baseline')
  const lo = Math.min(...rows.map((r) => r.valMean - r.valSd))
  const hi = Math.max(...rows.map((r) => r.valMean + r.valSd))
  const pad = Math.max((hi - lo) * 0.15, 0.0005)
  const x0 = lo - pad
  const x1 = hi + pad
  const LABEL = width < 480 ? 104 : 140
  const VALUE = width < 480 ? 0 : 128
  const ROW = 30
  const TOP = 8
  const AXIS = 34
  const plotW = Math.max(60, width - LABEL - VALUE - 8)
  const sx = (v: number) => LABEL + ((v - x0) / (x1 - x0)) * plotW
  const h = TOP + rows.length * ROW + AXIS
  const ticks = 4
  return (
    <div ref={ref} className="augdots">
      {width > 0 && (
        <svg width={width} height={h} role="img" aria-labelledby={titleId}>
          <title id={titleId}>{label}</title>
          {Array.from({ length: ticks + 1 }, (_, i) => {
            const v = x0 + ((x1 - x0) * i) / ticks
            return (
              <g key={i}>
                <line x1={sx(v)} x2={sx(v)} y1={TOP} y2={TOP + rows.length * ROW} className="grid-line" />
                <text x={sx(v)} y={TOP + rows.length * ROW + 16} textAnchor="middle" className="axis-text">
                  {pct(v, 1)}
                </text>
              </g>
            )
          })}
          <text x={LABEL + plotW / 2} y={h - 3} textAnchor="middle" className="axis-title">
            {t('results.aug.axis')}
          </text>
          {base && (
            <g>
              <line x1={sx(base.valMean)} x2={sx(base.valMean)} y1={TOP - 4} y2={TOP + rows.length * ROW} className="augdots__base" />
            </g>
          )}
          {rows.map((r, i) => {
            const y = TOP + i * ROW + ROW / 2
            const cls = `augdots__row is-${r.verdict}${r.selected ? ' is-selected' : ''}`
            return (
              <g key={r.name} className={cls}>
                <text x={LABEL - 10} y={y + 4} textAnchor="end" className="axis-text augdots__name">
                  {condName(t, r.name)}
                </text>
                <line x1={sx(r.valMean - r.valSd)} x2={sx(r.valMean + r.valSd)} y1={y} y2={y} className="augdots__sd" />
                {r.verdict === 'hurts' ? (
                  <rect x={sx(r.valMean) - 4.5} y={y - 4.5} width={9} height={9} className="augdots__dot" />
                ) : r.verdict === 'helps' ? (
                  <path d={`M${sx(r.valMean)} ${y - 6}L${sx(r.valMean) + 6} ${y}L${sx(r.valMean)} ${y + 6}L${sx(r.valMean) - 6} ${y}Z`} className="augdots__dot" />
                ) : (
                  <circle cx={sx(r.valMean)} cy={y} r={4.5} className="augdots__dot" />
                )}
                {VALUE > 0 && (
                  <text x={width - 4} y={y + 4} textAnchor="end" className="axis-text num">
                    {`${pct(r.valMean, 2)} ± ${(r.valSd * 100).toFixed(2)}`}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}

function AugStudyView({ study }: { study: AugStudy }) {
  const { t } = useI18n()
  const rows = augRows(study)
  const stress = augStressKeys(study)
  const base = rows.find((r) => r.name === 'baseline')
  const model = isModelKind(study.model) ? t(`model.${study.model}`) : study.model
  const robustCols = ['clean', ...stress]
  const value = (r: AugRow, col: string) => (col === 'clean' ? r.testMean : r.robust[col])
  const bestOf = (col: string) => Math.max(...rows.map((r) => value(r, col)).filter(finite))
  // shade is relative to this table's own range, so differences of a few points stay visible
  const all = rows.flatMap((r) => robustCols.map((c) => value(r, c))).filter(finite)
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  const shade = (v: number) => (hi > lo ? 0.06 + 0.34 * ((v - lo) / (hi - lo)) : 0.2)
  const transformNames = Object.keys(study.defaults ?? {})
  const [imgOk, setImgOk] = useState(false)

  return (
    <>
      <p className="caption">
        {t('results.aug.setup', {
          model,
          seeds: study.seeds.length,
          k: study.k_copies,
          selected: condName(t, study.selected_by_validation),
        })}
      </p>
      <ChartFigure
        title={t('results.aug.chartTitle', { model })}
        caption={t('results.aug.chartNote')}
        legend={[
          { label: t('results.aug.legend.helps'), color: 'var(--ok)', marker: 'diamond' },
          { label: t('results.aug.legend.unclear'), color: 'var(--ink-2)', marker: 'circle' },
          { label: t('results.aug.legend.hurts'), color: 'var(--danger)', marker: 'square' },
        ]}
        table={{
          columns: [t('results.aug.col.condition'), t('results.aug.col.val'), t('results.aug.col.delta'), t('results.aug.col.effect')],
          rows: rows.map((r) => [condName(t, r.name), `${pct(r.valMean, 2)} ± ${(r.valSd * 100).toFixed(2)}`, `${pts(r.delta)} ± ${finite(r.deltaSd) ? (r.deltaSd * 100).toFixed(2) : '–'}`, t(VERDICT_KEY[r.verdict])]),
        }}
      >
        <AugDots rows={rows} label={t('results.aug.aria', { model })} />
      </ChartFigure>

      <div className="table-wrap" role="region" aria-label={t('results.aug.effectRegion')} tabIndex={0}>
        <table className="table num">
          <caption className="visually-hidden">{t('results.aug.effectCaption')}</caption>
          <thead>
            <tr>
              <th scope="col" className="is-left">
                {t('results.aug.col.condition')}
              </th>
              <th scope="col">{t('results.aug.col.val')}</th>
              <th scope="col">{t('results.aug.col.delta')}</th>
              <th scope="col" className="is-left">
                {t('results.aug.col.effect')}
              </th>
              <th scope="col">{t('results.aug.col.test')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className={r.selected ? 'is-chosen' : undefined}>
                <th scope="row" className="is-left">
                  {condName(t, r.name)}
                  {r.selected && <span className="tag">{t('results.aug.selected')}</span>}
                </th>
                <td>
                  {pct(r.valMean, 2)}
                  <span className="td-sub">{`± ${(r.valSd * 100).toFixed(2)}`}</span>
                </td>
                <td>
                  {r.name === 'baseline' ? '–' : pts(r.delta)}
                  {r.name !== 'baseline' && <span className="td-sub">{`± ${finite(r.deltaSd) ? (r.deltaSd * 100).toFixed(2) : '–'}`}</span>}
                </td>
                <td className={`is-left verdict-${r.verdict}`}>{t(VERDICT_KEY[r.verdict])}</td>
                <td>
                  {pct(r.testMean, 2)}
                  <span className="td-sub">{`± ${(r.testSd * 100).toFixed(2)}`}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">{t('results.aug.rule', { n: study.seeds.length })}</p>
      <p className="caption">{t('results.aug.selection')}</p>

      {stress.length > 0 && (
        <div className="rob">
          <h3 className="chart__title">{t('results.aug.robTitle')}</h3>
          <p className="caption">{t('results.aug.robNote')}</p>
          <div className="table-wrap" role="region" aria-label={t('results.aug.robTitle')} tabIndex={0}>
            <table className="table num table--heat">
              <caption className="visually-hidden">{t('results.aug.robCaption')}</caption>
              <thead>
                <tr>
                  <th scope="col" className="is-left">
                    {t('results.aug.col.trainedWith')}
                  </th>
                  {robustCols.map((c) => (
                    <th key={c} scope="col">
                      {c === 'clean' ? t('results.aug.clean') : condName(t, c)}
                      {c !== 'clean' && study.stress?.[c] != null && <span className="th-sub">{strength(c, study.stress[c])}</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.name}>
                    <th scope="row" className="is-left">
                      {condName(t, r.name)}
                    </th>
                    {robustCols.map((c) => {
                      const v = value(r, c)
                      const isBest = finite(v) && v === bestOf(c)
                      const baseV = base ? value(base, c) : null
                      return (
                        <td key={c} className={isBest ? 'is-best' : undefined} style={{ '--heat-o': finite(v) ? shade(v) : 0 } as CSSProperties}>
                          {pct(v, 1)}
                          {r.name !== 'baseline' && finite(v) && finite(baseV) && <span className="td-sub">{pts(v - baseV, 1)}</span>}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {transformNames.length > 0 && (
        <div className="table-wrap" role="region" aria-label={t('results.aug.rangesTitle')} tabIndex={0}>
          <table className="table num">
            <caption className="caption table-caption">{t('results.aug.rangesTitle')}</caption>
            <thead>
              <tr>
                <th scope="col" className="is-left">
                  {t('results.aug.col.transform')}
                </th>
                <th scope="col">{t('results.aug.col.trainRange')}</th>
                <th scope="col">{t('results.aug.col.stress')}</th>
              </tr>
            </thead>
            <tbody>
              {transformNames.map((n) => (
                <tr key={n}>
                  <th scope="row" className="is-left">
                    {condName(t, n)}
                  </th>
                  <td>{strength(n, study.defaults[n])}</td>
                  <td>{strength(n, study.stress?.[n])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <figure className="augimg" hidden={!imgOk}>
        <img src={`${import.meta.env.BASE_URL}data/augmentation_examples.png`} alt={t('results.aug.imgAlt')} loading="lazy" onLoad={() => setImgOk(true)} onError={() => setImgOk(false)} />
        <figcaption className="caption">{t('results.aug.imgCaption')}</figcaption>
      </figure>
    </>
  )
}

export function AugmentationSection({ studies }: { studies: NonNullable<Results['augmentation']> }) {
  const { t } = useI18n()
  const kinds = (['cnn', 'mlp'] as const).filter((k) => studies[k])
  const [kind, setKind] = useState<'cnn' | 'mlp'>(kinds[0])
  const study = studies[kind] ?? studies[kinds[0]]!
  return (
    <Section
      id="r-aug"
      title={t('results.aug.title')}
      intro={
        <>
          <p>{t('results.aug.intro')}</p>
          {kinds.length > 1 && <Segmented legend={t('results.cm.model')} value={kind} onChange={setKind} options={kinds.map((k) => ({ value: k, label: t(`model.${k}`) }))} />}
        </>
      }
    >
      <AugStudyView key={kind} study={study} />
    </Section>
  )
}

// --------------------------------------------------------------------------- CNN vs MLP
interface Metric {
  key: string
  label: string
  get: (r: CnnVsMlpRow) => number | null | undefined
  fmt: (v: number) => string
  higher: boolean
  log?: boolean
}

export function CnnVsMlpSection({ data }: { data: CnnVsMlp }) {
  const { t } = useI18n()
  const rows = [...data.rows].sort((a, b) => MODEL_ORDER.indexOf(a.id) - MODEL_ORDER.indexOf(b.id))
  const robustKeys = [...new Set(rows.flatMap((r) => Object.keys(r.robust ?? {})))]
  const base: Metric[] = [
    { key: 'test', label: t('results.cvm.testAcc'), get: (r) => r.accuracy?.test, fmt: (v) => pct(v, 2), higher: true },
    { key: 'val', label: t('results.cvm.valAcc'), get: (r) => r.accuracy?.val, fmt: (v) => pct(v, 2), higher: true },
    { key: 'train', label: t('results.cvm.trainAcc'), get: (r) => r.accuracy?.train, fmt: (v) => pct(v, 2), higher: true },
    { key: 'gap', label: t('results.cvm.gap'), get: (r) => r.generalization_gap, fmt: (v) => `${(v * 100).toFixed(2)} ${t('results.cvm.points')}`, higher: false },
    { key: 'f1', label: t('results.cvm.f1'), get: (r) => r.f1_macro, fmt: (v) => num(v, 4), higher: true },
    { key: 'prec', label: t('results.cvm.precision'), get: (r) => r.precision_macro, fmt: (v) => num(v, 4), higher: true },
    { key: 'rec', label: t('results.cvm.recall'), get: (r) => r.recall_macro, fmt: (v) => num(v, 4), higher: true },
    { key: 'params', label: t('results.cvm.params'), get: (r) => r.n_parameters, fmt: (v) => int(v), higher: false, log: true },
    { key: 'bytes', label: t('results.cvm.bytes'), get: (r) => r.onnx_bytes, fmt: (v) => bytes(v), higher: false, log: true },
    { key: 'trainTime', label: t('results.cvm.trainTime'), get: (r) => r.train_time_sec, fmt: (v) => seconds(v), higher: false, log: true },
    { key: 'epochs', label: t('results.cvm.epochs'), get: (r) => r.epochs, fmt: (v) => int(v), higher: false },
    { key: 'latency', label: t('results.cvm.latency'), get: (r) => r.latency_ms, fmt: (v) => ms(v), higher: false, log: true },
    { key: 'pred10k', label: t('results.cvm.pred10k'), get: (r) => r.predict_time_10k_sec, fmt: (v) => seconds(v), higher: false, log: true },
  ]
  const robust: Metric[] = robustKeys.map((k) => ({ key: `rob-${k}`, label: t('results.cvm.robust', { name: condName(t, k) }), get: (r: CnnVsMlpRow) => r.robust?.[k], fmt: (v: number) => pct(v, 1), higher: true }))
  const metrics = [...base, ...robust].filter((m) => rows.some((r) => finite(m.get(r))))
  const multiples = metrics.filter((m) => ['test', 'gap', 'params', 'latency', 'trainTime', ...robustKeys.map((k) => `rob-${k}`)].includes(m.key))
  const mc = data.mcnemar

  return (
    <Section id="r-cvm" title={t('results.cvm.title')} intro={<p>{t('results.cvm.intro')}</p>}>
      <div className="table-wrap" role="region" aria-label={t('results.cvm.title')} tabIndex={0}>
        <table className="table num">
          <caption className="visually-hidden">{t('results.cvm.caption')}</caption>
          <thead>
            <tr>
              <th scope="col" className="is-left">
                {t('results.cvm.metric')}
              </th>
              {rows.map((r) => (
                <th key={r.id} scope="col">
                  {t(`model.${r.id}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => {
              const vals = rows.map((r) => m.get(r))
              const ok = vals.filter(finite)
              const best = ok.length > 1 ? (m.higher ? Math.max(...ok) : Math.min(...ok)) : null
              return (
                <tr key={m.key}>
                  <th scope="row" className="is-left">
                    {m.label}
                  </th>
                  {vals.map((v, i) => (
                    <td key={rows[i].id} className={finite(v) && v === best ? 'is-best' : undefined}>
                      {finite(v) ? m.fmt(v) : '–'}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="caption">{t('results.cvm.bestNote')}</p>

      <div className="multiples" role="list" aria-label={t('results.cvm.multiples')}>
        {multiples.map((m) => {
          const vals = rows.map((r) => m.get(r))
          const max = Math.max(...vals.filter(finite))
          const min = Math.min(...vals.filter(finite))
          return (
            <div key={m.key} className="multiple" role="listitem">
              <p className="multiple__title">{m.label}</p>
              <ul>
                {rows.map((r, i) => {
                  const v = vals[i]
                  // log metrics: bar length on a log scale from a tenth of the smallest value
                  const frac = !finite(v) || max <= 0 ? 0 : m.log ? Math.log(v / (min / 10)) / Math.log(max / (min / 10)) : v / max
                  return (
                    <li key={r.id} className={`multiple__row is-${r.id}`}>
                      <span className="multiple__name">{t(`model.${r.id}`)}</span>
                      <span className="multiple__track" aria-hidden="true">
                        <span className="multiple__fill" style={{ transform: `scaleX(${Math.max(0, Math.min(1, frac))})` }} />
                      </span>
                      <span className="multiple__val num">{finite(v) ? m.fmt(v) : '–'}</span>
                    </li>
                  )
                })}
              </ul>
              {m.log && <p className="multiple__note">{t('results.cvm.logNote')}</p>}
            </div>
          )
        })}
      </div>

      {mc && (
        <p className="caption">
          {t('results.cvm.mcnemar', { mlp: int(mc.only_mlp), cnn: int(mc.only_cnn), p: mc.p < 1e-4 ? mc.p.toExponential(1) : mc.p.toFixed(4) })}
        </p>
      )}
      {data.interpretability_note && (
        <div className="quote">
          <p className="label">{t('results.cvm.noteTitle')}</p>
          <p className="caption">{data.interpretability_note}</p>
        </div>
      )}
    </Section>
  )
}

// --------------------------------------------------------------------------- cross-dataset
export function CrossEvalSection({ dataset, entries }: { dataset: string; entries: CrossEval[] }) {
  const { t } = useI18n()
  const { manifest, results } = useBundle()
  const here = manifest.datasets[dataset]?.title ?? dataset
  const own = results[dataset]?.models ?? []
  const title = (id: string) => manifest.datasets[id]?.title ?? id
  const modelLabel = (m: string) => (isModelKind(m) ? t(`model.${m}`) : m)
  return (
    <Section id="r-cross" title={t('results.cross.title')} intro={<p>{t('results.cross.intro', { here })}</p>}>
      {entries.map((e) => (
        <div key={e.trained_on} className="cross">
          <p className="cross__badge">{t('results.cross.badge', { here })}</p>
          <h3 className="chart__title">{t('results.cross.heading', { from: title(e.trained_on), here })}</h3>
          <div className="table-wrap" role="region" aria-label={t('results.cross.heading', { from: title(e.trained_on), here })} tabIndex={0}>
            <table className="table num">
              <thead>
                <tr>
                  <th scope="col" className="is-left">
                    {t('results.metrics.model')}
                  </th>
                  <th scope="col">{t('results.cross.colOther', { from: title(e.trained_on) })}</th>
                  <th scope="col">{t('results.cross.colOwn', { here })}</th>
                </tr>
              </thead>
              <tbody>
                {e.rows.map((r) => {
                  const mine = own.find((m) => m.id === r.model)?.splits?.test?.accuracy
                  return (
                    <tr key={r.model}>
                      <th scope="row" className="is-left">
                        {modelLabel(r.model)}
                      </th>
                      <td>{pct(r.accuracy, 2)}</td>
                      <td>{pct(mine, 2)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {e.note && <p className="caption">{e.note}</p>}
        </div>
      ))}
    </Section>
  )
}

// --------------------------------------------------------------------------- all datasets
export function AllDatasets() {
  const { t } = useI18n()
  const { manifest, results } = useBundle()
  const ids = datasetIdsOf(manifest)
  const matrix = datasetMatrix(results, ids, MODEL_ORDER)
  const spreads = datasetSpreads(matrix)
  const title = (id: string) => manifest.datasets[id].title
  const bestAcc = (ds: string) => Math.max(...matrix.models.map((m) => matrix.cells[ds]?.[m]?.accuracy ?? -1))
  const bestF1 = (ds: string) => Math.max(...matrix.models.map((m) => matrix.cells[ds]?.[m]?.f1 ?? -1))
  return (
    <section className="rsection" aria-labelledby="r-all">
      <div className="rsection__head">
        <h2 id="r-all" className="h2">
          {t('results.all.title')}
        </h2>
        <div className="rsection__intro">
          <p>{t('results.all.intro')}</p>
        </div>
      </div>
      <div className="rsection__body">
        <div className="table-wrap" role="region" aria-label={t('results.all.title')} tabIndex={0}>
          <table className="table num">
            <caption className="visually-hidden">{t('results.all.caption')}</caption>
            <thead>
              <tr>
                <th scope="col" rowSpan={2} className="is-left">
                  {t('results.metrics.model')}
                </th>
                {ids.map((id) => (
                  <th key={id} scope="colgroup" colSpan={2} className="is-group">
                    {title(id)}
                  </th>
                ))}
              </tr>
              <tr>
                {ids.flatMap((id) => [
                  <th key={`${id}-a`} scope="col">
                    {t('results.all.acc')}
                  </th>,
                  <th key={`${id}-f`} scope="col">
                    {t('results.all.f1')}
                  </th>,
                ])}
              </tr>
            </thead>
            <tbody>
              {matrix.models.map((m) => (
                <tr key={m}>
                  <th scope="row" className="is-left">
                    {t(`model.${m}`)}
                  </th>
                  {ids.flatMap((id) => {
                    const c = matrix.cells[id]?.[m]
                    return [
                      <td key={`${id}-a`} className={c?.accuracy != null && c.accuracy === bestAcc(id) ? 'is-best' : undefined}>
                        {pct(c?.accuracy, 2)}
                      </td>,
                      <td key={`${id}-f`} className={c?.f1 != null && c.f1 === bestF1(id) ? 'is-best' : undefined}>
                        {num(c?.f1, 4)}
                      </td>,
                    ]
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption">{t('results.all.note')}</p>
        <ul className="bullets">
          {spreads.map((s) =>
            s.best && s.worst ? (
              <li key={s.dataset}>
                {s.gapPoints != null
                  ? t('results.all.spread', {
                      title: title(s.dataset),
                      best: t(`model.${s.best.model}`),
                      bestAcc: pct(s.best.accuracy, 2),
                      worst: t(`model.${s.worst.model}`),
                      worstAcc: pct(s.worst.accuracy, 2),
                      gap: s.gapPoints.toFixed(2),
                    })
                  : t('results.all.single', { title: title(s.dataset), best: t(`model.${s.best.model}`), bestAcc: pct(s.best.accuracy, 2) })}
              </li>
            ) : (
              <li key={s.dataset}>{t('results.all.none', { title: title(s.dataset) })}</li>
            ),
          )}
        </ul>
      </div>
    </section>
  )
}

// --------------------------------------------------------------------------- plain vs augmented (one pair per model)
/** Recipe of the augmented variants in a few words: "rotation ±30°, shift ±2 px, scaling 0.9–1.1, 2 copies". */
function recipeLabel(t: T, recipe: Pairs['recipe']): string | null {
  if (!recipe) return null
  const parts: string[] = []
  for (const op of recipe.ops) {
    const v = recipe.ranges[op]
    if (op === 'rotation' && finite(v)) parts.push(t('results.pairs.rotation', { a: num(v, 0) }))
    else if (op === 'translation' && finite(v)) parts.push(t('results.pairs.translation', { a: num(v, 0) }))
    else if (op === 'scaling' && Array.isArray(v) && v.length === 2) parts.push(t('results.pairs.scaling', { lo: num(v[0], 1), hi: num(v[1], 1) }))
    else parts.push(condName(t, op))
  }
  if (finite(recipe.copies)) parts.push(t('results.pairs.copies', { n: recipe.copies! }))
  return t('results.pairs.recipe', { parts: parts.join(', ') })
}

const PAIR_ROT = [30, 90]

export function PairsSection({ pairs }: { pairs: Pairs }) {
  const { t } = useI18n()
  const kinds = MODEL_ORDER.filter((k) => pairs.models[k])
  const rot = PAIR_ROT.filter((a) => kinds.every((k) => finite(pairs.models[k]!.plain.test_rotation_acc[String(a)])))
  const recipe = recipeLabel(t, pairs.recipe)
  const better = (a: number | null | undefined, b: number | null | undefined, lowerIsBetter = false) =>
    finite(a) && finite(b) && a !== b ? ((lowerIsBetter ? a < b : a > b) ? 'is-best' : undefined) : undefined
  const sub = [t('results.pairs.plain'), t('results.pairs.aug')]
  const groups = [t('results.pairs.test'), ...rot.map((a) => t('results.pairs.rot', { a }))]
  const angles = pairs.rotation_angles
  const name = (k: (typeof kinds)[number]) => t(`model.${k}`)
  const PLAIN = 'var(--ink-3)'
  const AUG = 'var(--accent)'
  return (
    <Section id="r-pairs" title={t('results.pairs.title')} intro={recipe ? <p>{recipe}</p> : null}>
      <div className="table-wrap" role="region" aria-label={t('results.pairs.region')} tabIndex={0}>
        <table className="table num table--sticky pairs-table">
          <thead>
            <tr>
              <th scope="col" rowSpan={2} className="is-left">
                {t('results.metrics.model')}
              </th>
              {groups.map((g, i) => (
                <th key={g} scope="colgroup" colSpan={i === 0 ? 3 : 2} className="is-group">
                  {g}
                </th>
              ))}
            </tr>
            <tr>
              {sub.map((h) => (
                <th key={`t-${h}`} scope="col">
                  {h}
                </th>
              ))}
              <th scope="col">{t('results.pairs.delta')}</th>
              {rot.flatMap((g) =>
                sub.map((h) => (
                  <th key={`${g}-${h}`} scope="col">
                    {h}
                  </th>
                )),
              )}
            </tr>
          </thead>
          <tbody>
            {kinds.map((k) => {
              const { plain: p, augmented: a } = pairs.models[k]!
              const d = (a.test_acc - p.test_acc) * 100
              return (
                <tr key={k}>
                  <th scope="row" className="is-left">
                    {name(k)}
                  </th>
                  <td className={better(p.test_acc, a.test_acc)}>{pct(p.test_acc, 2)}</td>
                  <td className={better(a.test_acc, p.test_acc)}>{pct(a.test_acc, 2)}</td>
                  <td className={d < 0 ? 'is-neg' : undefined}>{`${d > 0 ? '+' : d < 0 ? '−' : ''}${num(Math.abs(d), 2)}`}</td>
                  {rot.flatMap((r) => {
                    const x = p.test_rotation_acc[String(r)]
                    const y = a.test_rotation_acc[String(r)]
                    return [
                      <td key={`${r}p`} className={better(x, y)}>
                        {pct(x, 1)}
                      </td>,
                      <td key={`${r}a`} className={better(y, x)}>
                        {pct(y, 1)}
                      </td>,
                    ]
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <ChartFigure
        title={t('results.pairs.chartTitle')}
        caption={t('results.pairs.chartNote')}
        legend={[
          { label: t('results.pairs.plain'), color: PLAIN, marker: 'circle', dashed: true },
          { label: t('results.pairs.aug'), color: AUG, marker: 'square' },
        ]}
        table={{
          columns: [t('results.metrics.model'), ...angles.flatMap((x) => [`±${x}° ${sub[0]}`, `±${x}° ${sub[1]}`])],
          rows: kinds.map((k) => [name(k), ...angles.flatMap((x) => [pct(pairs.models[k]!.plain.test_rotation_acc[String(x)], 1), pct(pairs.models[k]!.augmented.test_rotation_acc[String(x)], 1)])]),
        }}
      >
        <div className="pairs-charts">
          {kinds.map((k) => {
            const pts = (side: PairSide) => angles.filter((x) => finite(side.test_rotation_acc[String(x)])).map((x) => ({ x, y: side.test_rotation_acc[String(x)] }))
            return (
              <figure key={k} className="pairs-charts__one">
                <figcaption className="label">{name(k)}</figcaption>
                <LineChart
                  series={[
                    { key: 'plain', label: t('results.pairs.plain'), color: PLAIN, marker: 'circle', dashed: true, points: pts(pairs.models[k]!.plain) },
                    { key: 'aug', label: t('results.pairs.aug'), color: AUG, marker: 'square', points: pts(pairs.models[k]!.augmented) },
                  ]}
                  xLabel={t('results.pairs.axis')}
                  yLabel={t('results.curves.panel.yAcc')}
                  xFormat={(v) => `${v}`}
                  yFormat={(v) => pct(v, 0)}
                  yMin={0.3}
                  yMax={1}
                  height={170}
                  ariaLabel={`${t('results.pairs.aria')}: ${name(k)}`}
                />
              </figure>
            )
          })}
        </div>
      </ChartFigure>
    </Section>
  )
}
