// Pinned model bundle tooling (no dependencies).
//   node scripts/bundle.mjs check   committed manifest, model-lock.json and the remote manifest agree
//   node scripts/bundle.mjs fetch   download the pinned revision into public/bundle, verifying every SHA-256
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const sha = (buf) => createHash('sha256').update(buf).digest('hex')
const fail = (msg) => {
  console.error(`bundle: ${msg}`)
  process.exit(1)
}

const lockPath = resolve(root, 'model-lock.json')
if (!existsSync(lockPath)) fail('web/model-lock.json is missing (write it with `python -m src upload --write-lock`)')
const lock = JSON.parse(readFileSync(lockPath, 'utf8'))
for (const k of ['repo', 'revision', 'bundle_id', 'manifest_sha256']) if (!lock[k]) fail(`model-lock.json has no "${k}"`)
if (!/^[0-9a-f]{40}$/.test(lock.revision)) fail('model-lock.json revision must be a full 40-character commit hash, not a branch name')
const base = `https://huggingface.co/${lock.repo}/resolve/${lock.revision}`

async function get(rel) {
  const res = await fetch(`${base}/${rel}`)
  if (!res.ok) fail(`${rel}: HTTP ${res.status} from ${base}`)
  return Buffer.from(await res.arrayBuffer())
}

const committed = readFileSync(resolve(root, 'public/data/manifest.json'))
if (sha(committed) !== lock.manifest_sha256) fail('public/data/manifest.json does not match manifest_sha256 in model-lock.json')

const cmd = process.argv[2]
const remote = await get('manifest.json')
if (sha(remote) !== lock.manifest_sha256) fail(`remote manifest at ${lock.revision} does not match model-lock.json`)
const manifest = JSON.parse(remote.toString('utf8'))
if (manifest.bundle_id !== lock.bundle_id) fail('remote bundle_id differs from model-lock.json')

if (cmd === 'check') {
  console.log(`bundle: OK  ${lock.repo}@${lock.revision.slice(0, 12)}  bundle_id=${lock.bundle_id}`)
} else if (cmd === 'fetch') {
  const out = resolve(root, 'public/bundle')
  const put = (rel, buf) => {
    const p = resolve(out, rel)
    mkdirSync(dirname(p), { recursive: true })
    writeFileSync(p, buf)
  }
  put('manifest.json', remote)
  const files = Object.entries(manifest.files_sha256)
  for (let i = 0; i < files.length; i += 4) {
    await Promise.all(
      files.slice(i, i + 4).map(async ([rel, want]) => {
        const buf = await get(rel)
        if (sha(buf) !== want) fail(`${rel}: SHA-256 differs from the manifest`)
        put(rel, buf)
      }),
    )
  }
  console.log(`bundle: fetched ${files.length} files from ${lock.repo}@${lock.revision.slice(0, 12)} into public/bundle`)
} else fail('usage: bundle.mjs check|fetch')
