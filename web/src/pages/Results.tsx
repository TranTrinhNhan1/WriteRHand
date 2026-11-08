import { useMemo, useState } from 'react'
import { Section } from '../components/Section'
import '../styles/results.css'
import { Segmented } from '../components/Segmented'
import { BarChart } from '../components/charts/BarChart'
import { ChartFigure, type LegendItem } from '../components/charts/ChartFigure'
import { ConfusionMatrix } from '../components/charts/ConfusionMatrix'
import { LineChart, type LineSeries } from '../components/charts/LineChart'
import { MODEL_ORDER } from '../engine/contract'
import { useBundle } from '../engine/react'
import { useI18n } from '../i18n'
import { useChoice } from '../lib/choice'
import { date, int, ms, num, paramName, paramValue, pct, seconds } from '../lib/format'
import { curvePanel, historyPanels, isValidMatrix, splitOf, sweepPanels, type SeriesRole } from '../lib/results'
import { datasetIdsOf } from '../lib/selection'
import { EXTRA_ORDER, SPLIT_IDS, type ResultId, type ResultModel, type SplitId } from '../lib/types'
import { finite } from '../lib/scale'
import { AllDatasets, AugmentationSection, CnnVsMlpSection, CrossEvalSection, PairsSection } from './resultsExtra'

const SPLIT_COLOR: Record<SplitId, string> = { train: 'var(--c-train)', val: 'var(--c-val)', test: 'var(--c-test)' }

const ROLE_STYLE: Record<SeriesRole, { color: string; marker: 'circle' | 'square' | 'diamond'; dashed?: boolean }> = {
  train: { color: 'var(--c-train)', marker: 'circle' },
  val: { color: 'var(--c-val)', marker: 'square' },
  holdout: { color: 'var(--c-val)', marker: 'diamond', dashed: true },
  neutral: { color: 'var(--ink-2)', marker: 'circle' },
}

/** Bolds the best value of each column so it is not carried by colour alone. */
function best(values: (number | null | undefined)[], higherIsBetter = true): number | null {
  const v = values.filter(finite)
  if (!v.length) return null
  return higherIsBetter ? Math.max(...v) : Math.min(...v)
}

const RESULT_ORDER: ResultId[] = [...MODEL_ORDER, ...EXTRA_ORDER]

/** Model name for a table row; a model measured in Python only carries a compact tag. */
export function ModelName({ m, mark }: { m: ResultModel; mark?: string }) {
  const { t } = useI18n()
  return (
    <>
      {t(`model.${m.id}`)}
      {mark && <sup className="fnmark">{mark}</sup>}
      {m.in_browser === false && <span className="offline-tag">{t('results.extra.offline')}</span>}
    </>
  )
}

function Cell({ v, b, text }: { v: number | null | undefined; b: number | null; text: string }) {
  return <td className={finite(v) && b != null && v === b ? 'is-best' : undefined}>{text}</td>
}

function MetricsTable({ models, split }: { models: ResultModel[]; split: SplitId }) {
  const { t } = useI18n()
  const splitName = t(`split.${split}`)
  const cols = [
    { key: 'accuracy', label: t('results.metrics.accuracy') },
    { key: 'precision_macro', label: t('results.metrics.precision') },
    { key: 'recall_macro', label: t('results.metrics.recall') },
    { key: 'f1_macro', label: t('results.metrics.f1') },
  ] as const
  const bests = cols.map((c) => best(models.map((m) => splitOf(m, split)?.[c.key])))
  return (
    <>
    <div className="table-wrap" role="region" aria-label={t('results.metrics.region', { split: splitName })} tabIndex={0}>
      <table className="table num">
        <caption className="visually-hidden">{t('results.metrics.caption', { split: splitName })}</caption>
        <thead>
          <tr>
            <th scope="col" className="is-left">
              {t('results.metrics.model')}
            </th>
            {cols.map((c) => (
              <th key={c.key} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.id}>
              <th scope="row" className="is-left">
                <ModelName m={m} mark={m.stacking && split !== 'test' ? '†' : undefined} />
              </th>
              {cols.map((c, i) => (
                <Cell key={c.key} v={splitOf(m, split)?.[c.key]} b={bests[i]} text={pct(splitOf(m, split)?.[c.key], 2)} />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    {split !== 'test' && models.some((m) => m.stacking) && <p className="caption table-note">† {t('results.extra.stackFit')}</p>}
    </>
  )
}

const ALL = '__all'

export default function Results() {
  const { t } = useI18n()
  const { manifest } = useBundle()
  const { dataset, setDataset } = useChoice()
  const [all, setAll] = useState(false)
  const ids = datasetIdsOf(manifest)
  const tab = all ? ALL : dataset
  const options = [...ids.map((id) => ({ value: id, label: manifest.datasets[id].title })), ...(ids.length > 1 ? [{ value: ALL, label: t('results.tabs.all') }] : [])]
  return (
    <div className="results sheet">
      <header className="page-head">
        <h1 className="display">{t('results.title')}</h1>
        <Segmented
          legend={t('results.tabs.legend')}
          value={tab}
          onChange={(v) => {
            if (v === ALL) setAll(true)
            else {
              setAll(false)
              setDataset(v)
            }
          }}
          options={options}
        />
      </header>
      {tab === ALL ? <AllDatasets /> : <DatasetResults key={dataset} dataset={dataset} />}
    </div>
  )
}

function DatasetResults({ dataset }: { dataset: string }) {
  const { t, lang } = useI18n()
  const bundle = useBundle()
  const entry = bundle.manifest.datasets[dataset]
  const extra = bundle.extra[dataset]
  const base = bundle.results[dataset] ?? { models: [] }
  const results = extra ? { ...base, models: [...base.models, ...extra.models], tuning: { ...base.tuning, ...extra.tuning } } : base
  const models = [...results.models].sort((a, b) => RESULT_ORDER.indexOf(a.id) - RESULT_ORDER.indexOf(b.id))
  const classes = entry.classes
  const hw = results.hardware
  const ds = results.dataset ?? entry.dataset

  const [split, setSplit] = useState<SplitId>('test')
  const [cmModel, setCmModel] = useState<ResultId>(models.find((m) => m.id === 'cnn')?.id ?? models.find((m) => m.id === 'mlp')?.id ?? models[0]?.id)
  const [cmSplit, setCmSplit] = useState<SplitId>('test')
  const [curveModel, setCurveModel] = useState<ResultId>(models.find((m) => m.history && Object.keys(m.history).length)?.id ?? models[0]?.id)
  const [tuneModel, setTuneModel] = useState<ResultId>(models.find((m) => results.tuning?.[m.id]?.sweeps)?.id ?? models[0]?.id)

  const splitOptions = SPLIT_IDS.map((s) => ({ value: s, label: t(`split.${s}`) }))
  const legendOf = (): LegendItem[] => SPLIT_IDS.map((s) => ({ label: t(`split.${s}`), color: SPLIT_COLOR[s] }))
  const modelOptions = models.map((m) => ({ value: m.id, label: t(`model.${m.id}`) }))
  const cm = models.find((m) => m.id === cmModel)
  const cmMatrix = cm ? splitOf(cm, cmSplit)?.confusion_matrix : null

  const accuracyRows = models.map((m) => ({ label: t(`model.${m.id}`), values: SPLIT_IDS.map((s) => splitOf(m, s)?.accuracy) }))
  const f1Rows = models.map((m) => ({ label: t(`model.${m.id}`), values: SPLIT_IDS.map((s) => splitOf(m, s)?.f1_macro) }))
  const series = SPLIT_IDS.map((s) => ({ label: t(`split.${s}`), color: SPLIT_COLOR[s] }))
  const tableOf = (rows: typeof accuracyRows) => ({
    columns: [t('results.metrics.model'), ...SPLIT_IDS.map((s) => t(`split.${s}`))],
    rows: rows.map((r) => [r.label, ...r.values.map((v) => pct(v, 2))]),
  })

  const nTest = ds?.n_test
  const timeRows = (pick: (m: ResultModel) => number | null | undefined) => models.map((m) => ({ label: t(`model.${m.id}`), values: [pick(m)] }))
  const trainTimes = timeRows((m) => m.train_time_sec)
  const batchTimes = timeRows((m) => {
    const t = splitOf(m, 'test')?.predict_time_sec
    return finite(t) ? t * 1000 : null
  })
  const latency = timeRows((m) => m.latency_ms_single_sample)

  const curveModelData = models.find((m) => m.id === curveModel)
  const panels = historyPanels(curveModel, curveModelData?.history)
  const tuning = results.tuning?.[tuneModel]
  // eslint-disable-next-line react-hooks/exhaustive-deps -- labels depend on the active language
  const sweeps = useMemo(() => sweepPanels(tuning), [tuning, lang])
  const curve = curvePanel(tuning)

  const sk = models.filter((m) => m.sklearn)
  const stack = models.find((m) => m.stacking)

  const modelName = (id: ResultId) => t(`model.${id}`)
  const unit = (v: number | null | undefined) => (finite(v) ? String(v!.toPrecision(4)) : '\u2013')
  const hwLine = [hw?.cpu ?? t('results.facts.unknownCpu'), hw?.logical_cores ? t('results.facts.cores', { n: hw.logical_cores }) : '', hw?.device ?? '']
    .filter(Boolean)
    .join(', ')

  if (models.length === 0) {
    return <p className="caption rsection">{t('results.noData', { title: entry.title })}</p>
  }

  return (
    <>
      <div className="results__intro">
        <dl className="facts">
          <div>
            <dt>{t('results.facts.dataset')}</dt>
            <dd>{entry.title}</dd>
          </div>
          <div>
            <dt>{t('results.facts.sizes')}</dt>
            <dd className="num">{t('results.facts.sizesValue', { train: int(ds?.n_train), val: int(ds?.n_val), test: int(ds?.n_test) })}</dd>
          </div>
          <div>
            <dt>{t('results.facts.measured')}</dt>
            <dd>{hwLine}</dd>
          </div>
          <div>
            <dt>{t('results.facts.generated')}</dt>
            <dd>
              {date(results.generated)}
              {results.git?.short ? `, ${t('results.facts.commit', { sha: results.git.short })}` : ''}
            </dd>
          </div>
        </dl>
      </div>

      <Section
        id="r-metrics"
        title={t('results.metrics.title')}
        intro={
          <>
            <p>{t('results.metrics.intro')}</p>
            <Segmented legend={t('split.label')} value={split} onChange={setSplit} options={splitOptions} />
          </>
        }
      >
        <MetricsTable models={models} split={split} />
        <div className="chart-pair">
          <ChartFigure title={t('results.metrics.accBy')} caption={t('results.metrics.accByNote')} legend={legendOf()} table={tableOf(accuracyRows)}>
            <BarChart
              rows={accuracyRows}
              series={series}
              scale="linear"
              min={0}
              max={1}
              axisLabel={t('results.metrics.accAxis')}
              format={(v) => pct(v, 2)}
              ariaLabel={t('results.metrics.accAria')}
            />
          </ChartFigure>
          <ChartFigure title={t('results.metrics.f1By')} legend={legendOf()} table={tableOf(f1Rows)}>
            <BarChart
              rows={f1Rows}
              series={series}
              scale="linear"
              min={0}
              max={1}
              axisLabel={t('results.metrics.f1Axis')}
              format={(v) => num(v, 4)}
              ariaLabel={t('results.metrics.f1Aria')}
            />
          </ChartFigure>
        </div>
      </Section>

      {extra?.pairs && <PairsSection pairs={extra.pairs} />}

      <Section id="r-time" title={t('results.time.title')} intro={<p>{t('results.time.intro')}</p>}>
        <div className="table-wrap" role="region" aria-label={t('results.time.region')} tabIndex={0}>
          <table className="table num">
            <thead>
              <tr>
                <th scope="col" className="is-left">
                  {t('results.time.model')}
                </th>
                <th scope="col">{t('results.time.training')}</th>
                <th scope="col">{nTest ? t('results.time.predictN', { n: int(nTest) }) : t('results.time.predictTest')}</th>
                <th scope="col">{t('results.time.perImage')}</th>
                <th scope="col">{t('results.time.oneImage')}</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const tm = splitOf(m, 'test')
                return (
                  <tr key={m.id}>
                    <th scope="row" className="is-left">
                      <ModelName m={m} mark={m.stacking ? '‡' : undefined} />
                    </th>
                    <td>{seconds(m.train_time_sec)}</td>
                    <td>{seconds(tm?.predict_time_sec)}</td>
                    <td>{ms(tm?.predict_time_ms_per_sample)}</td>
                    <td>{ms(m.latency_ms_single_sample)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {stack?.stacking && (
          <p className="caption table-note">
            ‡ {t('results.extra.stackCost', { base: seconds(stack.stacking.base_fit_sec), meta: seconds(stack.stacking.meta_fit_sec) })}
          </p>
        )}
        <p className="caption">
          {[
            hw?.cpu ?? t('results.time.noCpu'),
            hw?.logical_cores ? t('results.facts.cores', { n: hw.logical_cores }) : '',
            hw?.torch_threads ? t('results.time.threads', { n: hw.torch_threads }) : '',
            hw?.python ? t('results.time.py', { v: hw.python }) : '',
            hw?.numpy ? t('results.time.np', { v: hw.numpy }) : '',
            hw?.torch ? t('results.time.torch', { v: hw.torch }) : '',
          ]
            .filter(Boolean)
            .join(', ')}
          .
        </p>
        <div className="chart-triple">
          <ChartFigure
            title={t('results.time.trainTitle')}
            caption={t('results.time.logScale')}
            table={{ columns: [t('results.time.model'), t('results.time.colSeconds')], rows: trainTimes.map((r) => [r.label, unit(r.values[0])]) }}
          >
            <BarChart
              rows={trainTimes}
              series={[{ label: t('results.time.series.train'), color: 'var(--c-train)' }]}
              scale="log"
              axisLabel={t('results.time.seconds')}
              format={seconds}
              ariaLabel={t('results.time.trainAria')}
            />
          </ChartFigure>
          <ChartFigure
            title={nTest ? t('results.time.predictTitle', { n: int(nTest) }) : t('results.time.predictTitleTest')}
            caption={t('results.time.logScale')}
            table={{ columns: [t('results.time.model'), t('results.time.colMs')], rows: batchTimes.map((r) => [r.label, unit(r.values[0])]) }}
          >
            <BarChart
              rows={batchTimes}
              series={[{ label: t('results.time.series.predict'), color: 'var(--c-val)' }]}
              scale="log"
              axisLabel={t('results.time.milliseconds')}
              format={(v) => ms(v)}
              ariaLabel={t('results.time.predictAria')}
            />
          </ChartFigure>
          <ChartFigure
            title={t('results.time.oneTitle')}
            caption={t('results.time.logScale')}
            table={{ columns: [t('results.time.model'), t('results.time.colMs')], rows: latency.map((r) => [r.label, unit(r.values[0])]) }}
          >
            <BarChart
              rows={latency}
              series={[{ label: t('results.time.series.latency'), color: 'var(--c-test)' }]}
              scale="log"
              axisLabel={t('results.time.milliseconds')}
              format={(v) => ms(v)}
              ariaLabel={t('results.time.oneAria')}
            />
          </ChartFigure>
        </div>
      </Section>

      <Section
        id="r-cm"
        title={t('results.cm.title')}
        intro={
          <>
            <Segmented legend={t('results.cm.model')} value={cmModel} onChange={setCmModel} options={modelOptions} />
            <Segmented legend={t('split.label')} value={cmSplit} onChange={setCmSplit} options={splitOptions} />
          </>
        }
      >
        {cm && isValidMatrix(cmMatrix) ? (
          <ConfusionMatrix matrix={cmMatrix} classes={classes} name={t('cm.name', { model: modelName(cm.id), split: t(`split.${cmSplit}`) })} />
        ) : (
          <p className="caption">{t('results.cm.none')}</p>
        )}
      </Section>

      <Section
        id="r-curves"
        title={t('results.curves.title')}
        intro={
          <>
            <p>{t('results.curves.intro')}</p>
            <Segmented legend={t('results.cm.model')} value={curveModel} onChange={setCurveModel} options={modelOptions} />
          </>
        }
      >
        {panels.length === 0 ? (
          <p className="caption">{t('results.curves.none', { name: modelName(curveModel) })}</p>
        ) : (
          <div className="chart-pair">
            {panels.map((p) => {
              const n = Math.max(...p.series.map((s) => s.values.length))
              const lines: LineSeries[] = p.series.map((s) => ({
                key: s.key,
                label: t(s.labelKey),
                ...ROLE_STYLE[s.role],
                points: s.values.map((y, i) => ({ x: i + 1, y })),
              }))
              const marks: { x: number; label: string }[] = []
              if (curveModelData?.best_epoch && p.id !== 'alpha') marks.push({ x: curveModelData.best_epoch, label: t('results.curves.bestEpoch') })
              const fmt = p.format === 'pct' ? (v: number) => pct(v, 1) : (v: number) => num(v, 3)
              const xLabel = curveModel === 'adaboost' ? t('results.curves.round') : t('results.curves.epoch')
              const title = t(p.titleKey)
              return (
                <ChartFigure
                  key={p.id}
                  title={title}
                  legend={lines.map((l) => ({ label: l.label, color: l.color, marker: l.marker, dashed: l.dashed }))}
                  caption={
                    p.id === 'loss' && curveModelData?.stopped_epoch
                      ? t('results.curves.earlyStop', {
                          stopped: curveModelData.stopped_epoch,
                          best: curveModelData.best_epoch ?? curveModelData.stopped_epoch,
                        })
                      : undefined
                  }
                  table={{
                    columns: [xLabel, ...lines.map((l) => l.label)],
                    rows: Array.from({ length: n }, (_, i) => [i + 1, ...p.series.map((s) => (finite(s.values[i]) ? fmt(s.values[i]) : '\u2013'))]),
                  }}
                >
                  <LineChart
                    series={lines}
                    xLabel={xLabel}
                    yLabel={t(p.yKey)}
                    yFormat={fmt}
                    xFormat={(v) => String(v)}
                    marks={marks}
                    ariaLabel={t('results.curves.aria', { title, name: modelName(curveModel) })}
                  />
                </ChartFigure>
              )
            })}
          </div>
        )}
      </Section>

      <Section
        id="r-tuning"
        title={t('results.tuning.title')}
        intro={
          <>
            <p>{t('results.tuning.intro')}</p>
            <Segmented legend={t('results.cm.model')} value={tuneModel} onChange={setTuneModel} options={modelOptions} />
          </>
        }
      >
        {tuning?.best_params && (
          <div className="chosen">
            <p className="label">
              {t('results.tuning.chosenTitle')}
              {finite(tuning.n_trials) ? <span className="caption"> · {t('results.tuning.trials', { n: tuning.n_trials! })}</span> : null}
            </p>
            <dl className="params">
              {Object.entries(tuning.best_params).map(([k, v]) => (
                <div key={k}>
                  <dt>{paramName(k)}</dt>
                  <dd className="num">{paramValue(v)}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
        {sweeps.length === 0 && !curve ? (
          <p className="caption">{t('results.tuning.none')}</p>
        ) : (
          <div className="chart-pair">
            {sweeps.map((sw) => {
              const train = sw.points.filter((p) => finite(p.train)).map((p) => ({ x: p.x, y: p.train! }))
              const val = sw.points.filter((p) => finite(p.val)).map((p) => ({ x: p.x, y: p.val! }))
              const lines: LineSeries[] = [
                { key: 'train', label: t('results.tuning.trainAcc'), ...ROLE_STYLE.train, points: train },
                { key: 'val', label: t('results.tuning.valAcc'), ...ROLE_STYLE.val, points: val },
              ]
              const cats = sw.kind === 'categorical' ? sw.points.map((p) => p.label) : undefined
              const pname = paramName(sw.param)
              return (
                <ChartFigure
                  key={sw.param}
                  title={t('results.tuning.accAgainst', { param: pname })}
                  caption={[sw.chosen ? t('results.tuning.chosen', { value: sw.chosen.label }) : '', tuneModel === 'stacking' ? t('results.extra.stackCv') : ''].filter(Boolean).join(' ') || undefined}
                  legend={lines.map((l) => ({ label: l.label, color: l.color, marker: l.marker }))}
                  table={{
                    columns: [pname, t('results.tuning.trainAcc'), t('results.tuning.valAcc')],
                    rows: sw.points.map((p) => [p.label, pct(p.train, 2), pct(p.val, 2)]),
                  }}
                >
                  <LineChart
                    series={lines}
                    xLabel={pname}
                    yLabel={t('results.curves.panel.yAcc')}
                    xScale={sw.log ? 'log' : 'linear'}
                    xCategories={cats}
                    xFormat={cats ? undefined : (v) => paramValue(v)}
                    yFormat={(v) => pct(v, 1)}
                    marks={sw.chosen ? [{ x: sw.chosen.x, label: t('results.tuning.mark') }] : []}
                    ariaLabel={t('results.tuning.aria', { param: pname, name: modelName(tuneModel) })}
                  />
                </ChartFigure>
              )
            })}
            {curve && (
              <ChartFigure
                title={t('results.tuning.accAgainst', { param: paramName(curve.param) })}
                caption={curve.chosen ? t('results.tuning.chosen', { value: curve.chosen }) : undefined}
                legend={[
                  ...(curve.train.length ? [{ label: t('results.tuning.trainAcc'), color: ROLE_STYLE.train.color, marker: ROLE_STYLE.train.marker }] : []),
                  { label: t('results.tuning.valAcc'), color: ROLE_STYLE.val.color, marker: ROLE_STYLE.val.marker },
                ]}
                table={{
                  columns: [paramName(curve.param), t('results.tuning.trainAcc'), t('results.tuning.valAcc')],
                  rows: curve.val.map((v, i) => [i + 1, pct(curve.train[i], 2), pct(v, 2)]),
                }}
              >
                <LineChart
                  series={[
                    ...(curve.train.length
                      ? [{ key: 'train', label: t('results.tuning.trainAcc'), ...ROLE_STYLE.train, points: curve.train.map((y, i) => ({ x: i + 1, y })) }]
                      : []),
                    { key: 'val', label: t('results.tuning.valAcc'), ...ROLE_STYLE.val, points: curve.val.map((y, i) => ({ x: i + 1, y })) },
                  ]}
                  xLabel={paramName(curve.param)}
                  yLabel={t('results.curves.panel.yAcc')}
                  xFormat={(v) => String(v)}
                  yFormat={(v) => pct(v, 1)}
                  marks={curve.chosen ? [{ x: curve.chosen, label: t('results.tuning.mark') }] : []}
                  ariaLabel={t('results.tuning.ariaPlain', { param: paramName(curve.param) })}
                />
              </ChartFigure>
            )}
          </div>
        )}
      </Section>

      <Section id="r-sk" title={t('results.sk.title')} intro={<p>{t('results.sk.intro')}</p>}>
        {sk.length === 0 ? (
          <p className="caption">{t('results.sk.none')}</p>
        ) : (
          <div className="table-wrap" role="region" aria-label={t('results.sk.region')} tabIndex={0}>
            <table className="table num table--sticky">
              <caption className="visually-hidden">{t('results.sk.caption')}</caption>
              <thead>
                <tr>
                  <th scope="col" rowSpan={2} className="is-left">
                    {t('results.time.model')}
                  </th>
                  <th scope="col" rowSpan={2} className="is-left">
                    {t('results.sk.estimator')}
                  </th>
                  <th scope="colgroup" colSpan={2} className="is-group">
                    {t('results.sk.accuracy')}
                  </th>
                  <th scope="colgroup" colSpan={2} className="is-group">
                    {t('results.sk.f1')}
                  </th>
                  <th scope="colgroup" colSpan={2} className="is-group">
                    {t('results.sk.training')}
                  </th>
                </tr>
                <tr>
                  {[0, 1, 2].flatMap((g) =>
                    [t('results.sk.ours'), t('results.sk.theirs')].map((h, i) => (
                      <th key={`${g}-${i}`} scope="col">
                        {h}
                      </th>
                    )),
                  )}
                </tr>
              </thead>
              <tbody>
                {sk.map((m) => {
                  const s = m.sklearn!
                  const tm = splitOf(m, 'test')
                  return (
                    <tr key={m.id}>
                      <th scope="row" className="is-left">
                        <ModelName m={m} />
                      </th>
                      <td className="is-left">{s.class ?? '\u2013'}</td>
                      <td>{pct(tm?.accuracy, 2)}</td>
                      <td>{pct(s.splits?.test?.accuracy, 2)}</td>
                      <td>{num(tm?.f1_macro, 4)}</td>
                      <td>{num(s.splits?.test?.f1_macro, 4)}</td>
                      <td>{seconds(m.train_time_sec)}</td>
                      <td>{seconds(s.train_time_sec)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {results.cnn_vs_mlp && <CnnVsMlpSection data={results.cnn_vs_mlp} />}
      {results.augmentation && <AugmentationSection studies={results.augmentation} />}
      {results.cross_eval && <CrossEvalSection dataset={dataset} entries={results.cross_eval} />}
    </>
  )
}
