# WriteRHand web

Static demo site for WriteRHand: draw (or upload) a handwritten digit and let four from-scratch classifiers
(Gaussian Naive Bayes, softmax regression, MLP, AdaBoost-SAMME) read it in the browser. It also explains the models
and shows the evaluation results.

React 18, TypeScript (strict), Vite, `onnxruntime-web` 1.30.0 (WebAssembly, single-threaded), hand-written SVG charts,
KaTeX (MathML output) for the equations. No backend.

## Language, fonts and design

* **Languages.** Vietnamese (default) and English, switched with the VI | EN control at the top right. The choice is
  kept in `localStorage` (inside try/catch, so blocked storage only means it is not remembered) and `<html lang>`
  follows it. Copy lives in two typed dictionaries, `src/i18n/en.ts` and `src/i18n/vi.ts`, with a tiny `t()` helper in
  `src/i18n/index.tsx` and no i18n library. Both are written by hand, not machine-translated. `tests/i18n.test.ts` fails
  if the two files differ in keys, list lengths or `{placeholders}`. Numbers use "." for decimals and a narrow
  no-break space for thousands (`60 000`) in both languages.
* **Look.** "Ink Lab": a dark, instrument-like interface with one lime accent, hairline rules and small radii, plus a
  light theme. The theme follows the system setting until you use the sun/moon button (the choice is kept in
  `localStorage`, inside try/catch). The drawing screen and the model maps stay dark in both themes, like a real
  instrument display.
* **Fonts.** Sans-serif and monospace only, self-hosted through Fontsource (no CDN): Geist Variable for text and
  headings, Geist Mono Variable for numbers, labels and readouts. Both ship a Vietnamese subset.
* **Equations.** KaTeX parses the TeX and emits MathML; the browser draws it with a subset of Noto Sans Math
  (`src/assets/fonts/NotoSansMath-subset.woff2`, about 41 KB, SIL Open Font License 1.1), so equations are sans-serif
  too and no KaTeX CSS or fonts are shipped.
* **Motion.** Springs and entry animations are CSS and only run under `prefers-reduced-motion: no-preference`.

## Run

```bash
cd web
npm ci
npm run dev          # http://localhost:5173, base path "/"
npm run typecheck
npm test
npm run build        # tsc -b && vite build, base path "/WriteRHand/"
npm run preview      # serves dist/ under /WriteRHand/
```

## Environment variables (build time)

| Variable               | Default                                                              | Meaning                                         |
| ---------------------- | -------------------------------------------------------------------- | ----------------------------------------------- |
| `VITE_BASE`            | `/WriteRHand/` for `build`, `/` for `dev`                            | Public base path (GitHub Pages project site)    |
| `VITE_MODEL_BASE_URL`  | production: the revision pinned in `model-lock.json`; dev: `${BASE_URL}bundle` | Where model files are fetched from (override for tests, e.g. `/bundle`) |

## The pinned model bundle

* `model-lock.json` (written by `uv run python -m src upload --write-lock`) pins the Hugging Face repo, the **commit
  revision**, the `bundle_id` and the SHA-256 of `manifest.json`. Production builds fetch
  `https://huggingface.co/<repo>/resolve/<revision>/...` only, never `main`.
* `public/data/manifest.json` and `public/data/<dataset>/results.json` are committed same-origin copies written by
  `uv run python -m src export --site`. At start-up the site compares the SHA-256 of the manifest with the lock and
  refuses to run on a mismatch. Every downloaded model file is checked against the hash in the manifest.
* `npm run bundle:check` verifies that the committed manifest, the lock and the remote manifest agree;
  `npm run bundle:fetch` downloads the pinned bundle into `public/bundle/` (git-ignored) with every file verified.
* The trace contract that the explorer draws from is in `../docs/TRACE_CONTRACT.md`.

## How inference and the explorer work

* `src/engine/` is the data layer: `contract.ts` validates the manifest and the per-model trace
  (`writerhand.trace/1`), `store.ts` loads models lazily and produces an `Execution` stamped with the bundle, dataset,
  model and an input fingerprint (a stale execution is never shown after the model or input changes), `reconstruct.ts`
  recomputes selected quantities from `params.bin` for comparison with the runtime output, `adaboost.ts` reads the
  binary ensemble and walks the real trees.
* ONNX models (Naive Bayes, softmax, MLP, CNN) run with `onnxruntime-web/wasm`; the runtime outputs requested are exactly
  those listed in `trace.runtime_outputs`. `ort.env.wasm.numThreads = 1` because GitHub Pages cannot send COOP/COEP headers.
* `src/components/explorer/` draws the diagrams and animations from the trace metadata only (no hard-coded architecture).
  Activations are shown as what a layer computed for this input; they are not presented as an explanation of the decision.
  The training replay shows only recorded checkpoints and metrics (`writerhand.replay/1`).
* Input to every model: 784 pixels in [0, 1]; the (p - mean) / (std + eps) standardisation is part of the graphs / the
  AdaBoost evaluator. Preprocessing of drawings (`src/lib/preprocess.ts`): ink bounding box, area-averaged resize so the
  longer side is 20 px, paste into 28 x 28, shift so the centre of mass is at (14, 14).

## Tests

* `npm test` (Vitest, Node): engine contract and Python parity for every dataset/model (ONNX Runtime Web, AdaBoost
  binary), reconstructions, stale-result guard, hash mismatch, explorer maths, preprocessing, results helpers.
  The bundle tests are skipped locally when `public/bundle` is absent, but **fail** when `REQUIRE_BUNDLE=1` (CI and
  release) so a missing bundle can never pass silently.
* `npm run test:e2e` (Playwright, Chromium): builds the site, serves it with the pinned bundle and drives it like a
  visitor (draw, choose a model, look inside, step the animation, change the input). Set `PW_CHROMIUM` to an existing
  browser binary to avoid a download.
