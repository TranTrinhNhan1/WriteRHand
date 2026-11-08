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

| Variable               | Default                                                                       | Meaning                                              |
| ---------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------- |
| `VITE_BASE`            | `/WriteRHand/` for `build`, `/` for `dev`                                     | Public base path (GitHub Pages project site)         |
| `VITE_MODEL_BASE_URL`  | production: `https://huggingface.co/TranTrinhNhan1111/WriteRHand-MNIST/resolve/main`; dev: `${BASE_URL}bundle` | Where `models/*` of the model bundle are fetched from |

`VITE_MODEL_BASE_URL` may also be a same-origin path such as `/WriteRHand/bundle`.

## Data

* `public/data/manifest.json` and `public/data/results.json` are committed, same-origin copies produced by the Python
  pipeline. All numbers on the Models, Results and About pages are read from them at run time; nothing is hard-coded.
  Replace them with the real files after the final training run.
* `public/bundle/` (git-ignored) holds a local copy of the model bundle for development and tests: `manifest.json`,
  `models/{naive_bayes,softmax,mlp}.onnx`, `models/adaboost.json`, `test_vectors.json`. In production the models are
  loaded from `VITE_MODEL_BASE_URL`. If `public/bundle/` exists when you run `npm run build`, Vite copies it into
  `dist/bundle/` as well; remove it before building if you do not want that.

## How inference works

* Models are loaded lazily on first selection (the default, MLP, starts loading when you first touch the canvas).
* ONNX models (Naive Bayes, softmax, MLP) run with `onnxruntime-web/wasm`. The runtime is a dynamic import and its
  `.wasm` file is emitted by Vite as a hashed asset, so it is served from the site itself under the configured base
  path (no CDN). `ort.env.wasm.numThreads = 1` because GitHub Pages cannot send COOP/COEP headers.
* AdaBoost is exported as JSON and evaluated by `src/lib/adaboost.ts`.
* "Look inside" (`src/components/inside/`): the ONNX files carry extra, optional outputs next to `probabilities`
  (MLP hidden activations, softmax weights, Naive Bayes class means and variances; see `src/export/onnx_export.py`),
  and `src/lib/trace.ts` walks every AdaBoost tree for the drawing. The views only draw these numbers; they never
  change a prediction. If a model file has no trace outputs the panel says so and everything else still works.
* Input to every model: 784 pixels in [0, 1], white digit on black. The (p - mean) / (std + eps) standardisation is part
  of the ONNX graphs / the AdaBoost evaluator.
* Preprocessing (`src/lib/preprocess.ts`, pure functions on a grayscale buffer): ink bounding box, area-averaged
  resize so the longer side is 20 px, paste into 28 x 28, shift (bilinear) so the centre of mass is at (14, 14).
  Uploads are inverted when the mean luminance is above 0.5, background-subtracted and contrast-stretched first.

## Tests

`npm test` runs Vitest in Node:

* `tests/preprocess.test.ts`: centring, scale, aspect ratio, empty input, invert detection, noise, determinism.
* `tests/parity.test.ts`: loads `test_vectors.json` and the models from `public/bundle` (or `PARITY_BUNDLE_DIR`) and
  checks that the TypeScript AdaBoost evaluator and the ONNX models (run with onnxruntime-web's WebAssembly backend
  inside Node) reproduce the Python outputs, plus the 28 -> 280 -> 28 round trip of real MNIST digits.
  The Python side stores probabilities rounded to six decimals, so the AdaBoost comparison tolerance is 5e-7
  (argmax must be identical); ONNX tolerance is 1e-3. These tests are **skipped when the bundle is not present**, so
  fetch the bundle first in CI (for example with the Hugging Face CLI into `web/public/bundle`).
* `tests/results.test.ts`, `tests/scale.test.ts`: data helpers and chart scales.
