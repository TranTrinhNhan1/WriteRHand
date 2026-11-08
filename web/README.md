# WriteRHand web

Static demo site for WriteRHand: draw (or upload) a handwritten digit, or pick a test image, and let five classifiers
written from scratch (Gaussian Naive Bayes, softmax regression, MLP, a small CNN, AdaBoost-SAMME), each trained on three
datasets (MNIST, Fashion-MNIST, EMNIST Digits), read it in the browser. It also explains the models, shows the
evaluation results and serves the course deliverables (report and slides PDFs) on the Documents page.

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
* **Fonts.** Sans-serif and monospace for the interface (equations: Computer Modern, below), self-hosted through Fontsource (no CDN): Geist Variable for text and
  headings, Geist Mono Variable for numbers, labels and readouts. Both ship a Vietnamese subset.
* **Equations.** KaTeX typesets the TeX as HTML in its Computer Modern fonts, with MathML alongside for screen
  readers. `src/styles/katex.css` is the package stylesheet reduced to the woff2 files of `katex/dist/fonts` (bundled by
  Vite, no CDN); KaTeX and its CSS load only with the Models page.
* **Motion.** Springs and entry animations are CSS and only run under `prefers-reduced-motion: no-preference`.

## Run

```bash
cd web
npm ci
npm run dev          # http://localhost:5173, base path "/"
npm run typecheck
npm test
npm run docs:prepare # copy ../report/main.pdf and ../slides/WriteRHand-slides.pdf to public/docs/ (needs pdfinfo)
npm run build        # docs:prepare, tsc -b, vite build; base path "/WriteRHand/"
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
* `public/data/extra_models.json` holds the measured numbers of the models that are not in the bundle (linear SVM,
  stacking: no ONNX, never run in the browser; shown on Results and Models only). It is written from `results/` by
  `uv run python -m src site-extra` (and by `export --site`); a pytest checks the committed file matches `results/`.
* `npm run bundle:check` verifies that the committed manifest, the lock and the remote manifest agree;
  `npm run bundle:fetch` downloads the pinned bundle into `public/bundle/` (git-ignored) with every file verified.
* The trace contract that the explorer draws from is in `../docs/TRACE_CONTRACT.md`.

## Documents and routing

* The Documents page (`#/documents`) links `docs/report.pdf` and `docs/slides.pdf` and reads `docs/docs.json` (file
  name, page count, size, SHA-256 per PDF). `scripts/docs.mjs prepare` writes them into `public/docs/` (git-ignored, so
  the PDFs are not stored twice); page counts come from `pdfinfo` (poppler-utils), and a missing source PDF or a
  missing `pdfinfo` stops the build. `npm run build` and the Playwright web server both run it first.
* `REPO_PUBLIC` in `src/lib/config.ts` says whether the GitHub repository is public; while it is `false` the page states
  that the repository is private.
* Routing uses the URL hash (`#/models`, `#/documents`): the server only ever sees the site root, so reloading or sharing
  a link to any page works on GitHub Pages without a `404.html` fallback.

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
* The MLP and CNN networks are drawn in 3D with three.js (`explorer/three/`; loaded on demand in its own chunk):
  layer widths, channels × H × W and the strongest a·w connections come from the trace and the current execution
  (`three/spec.ts`). Without WebGL, or on request, the same panel shows the 2D drawing.
* Input to every model: 784 pixels in [0, 1]; the (p - mean) / (std + eps) standardisation is part of the graphs / the
  AdaBoost evaluator. Preprocessing of drawings (`src/lib/preprocess.ts`): ink bounding box, area-averaged resize so the
  longer side is 20 px, paste into 28 x 28, shift so the centre of mass is at (14, 14).

## Tests

* `npm test` (Vitest, Node): engine contract and Python parity for every dataset/model (ONNX Runtime Web, AdaBoost
  binary), reconstructions, stale-result guard, hash mismatch, explorer maths, preprocessing, results helpers.
  The bundle tests are skipped locally when `public/bundle` is absent, but **fail** when `REQUIRE_BUNDLE=1` (CI and
  release) so a missing bundle can never pass silently.
* `npm run test:e2e` (Playwright, Chromium): builds the site, serves it with the pinned bundle and drives it like a
  visitor (draw, choose a model, look inside, step the animation, change the input). `e2e/documents.spec.ts`
  checks the header GitHub link, the Documents page, the served PDFs against `docs.json`, the phone layout, reloading
  `#/documents`, and that no request fails. Set `PW_CHROMIUM` to an existing
  browser binary to avoid a download.
