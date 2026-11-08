#!/usr/bin/env node
/**
 * Publishes the course deliverables with the site.
 *
 *   node scripts/docs.mjs prepare
 *
 * copies ../report/main.pdf -> public/docs/report.pdf and ../slides/WriteRHand-slides.pdf -> public/docs/slides.pdf,
 * then writes public/docs/docs.json with the size, SHA-256 and page count of each file. Page counts come from
 * `pdfinfo` (poppler-utils). A missing source PDF, a missing pdfinfo or an unreadable page count is a hard error:
 * the site never publishes an empty Documents page or an invented count. public/docs/ is generated (git-ignored).
 */
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const ROOT = resolve(WEB, '..')
const OUT = join(WEB, 'public', 'docs')

const DOCS = {
  report: { src: join(ROOT, 'report', 'main.pdf'), file: 'report.pdf' },
  slides: { src: join(ROOT, 'slides', 'WriteRHand-slides.pdf'), file: 'slides.pdf' },
}

function fail(msg) {
  console.error(`docs: ${msg}`)
  process.exit(1)
}

/** The number from the `Pages:` line of `pdfinfo` output. */
function parsePages(text) {
  const m = /^Pages:\s+(\d+)\s*$/m.exec(text)
  return m ? Number(m[1]) : null
}

function pageCount(path) {
  let out
  try {
    out = execFileSync('pdfinfo', [path], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
  } catch (e) {
    if (e && e.code === 'ENOENT') fail('pdfinfo was not found. Install poppler-utils (apt install poppler-utils, brew install poppler).')
    fail(`pdfinfo could not read ${relative(ROOT, path)}: ${String((e && e.stderr) || e).trim()}`)
  }
  const pages = parsePages(out)
  if (!pages || pages < 1) fail(`pdfinfo gave no page count for ${relative(ROOT, path)}`)
  return pages
}

function prepare() {
  for (const { src } of Object.values(DOCS)) {
    if (!existsSync(src)) fail(`${relative(ROOT, src)} is missing. Build the report and the slides first (see the repository README).`)
  }
  rmSync(OUT, { recursive: true, force: true })
  mkdirSync(OUT, { recursive: true })
  const meta = {}
  for (const [id, { src, file }] of Object.entries(DOCS)) {
    // every number describes the published copy, so a source rebuilt during this run can never mismatch its metadata
    const dest = join(OUT, file)
    copyFileSync(src, dest)
    const buf = readFileSync(dest)
    if (buf.subarray(0, 5).toString('latin1') !== '%PDF-') fail(`${relative(ROOT, src)} is not a PDF file`)
    const pages = pageCount(dest)
    meta[id] = { file, pages, bytes: statSync(dest).size, sha256: createHash('sha256').update(buf).digest('hex') }
  }
  meta.generated = new Date().toISOString()
  writeFileSync(join(OUT, 'docs.json'), `${JSON.stringify(meta, null, 2)}\n`)
  for (const id of Object.keys(DOCS)) {
    const d = meta[id]
    console.log(`docs: ${d.file}  ${d.pages} pages  ${d.bytes} bytes  sha256 ${d.sha256.slice(0, 12)}`)
  }
}

if (process.argv[2] === 'prepare') prepare()
else fail('usage: node scripts/docs.mjs prepare')
