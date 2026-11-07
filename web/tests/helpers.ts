import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'

/** Directory with manifest.json, models/ and test_vectors.json (not committed; see README). */
export const BUNDLE_DIR = resolve(process.env.PARITY_BUNDLE_DIR ?? resolve(__dirname, '../public/bundle'))
export const hasBundle = existsSync(resolve(BUNDLE_DIR, 'test_vectors.json'))

export interface TestVectors {
  indices: number[]
  labels: number[]
  pixels: number[][]
  expected: Record<string, { pred: number[]; proba: number[][] }>
}

export function readJson<T>(rel: string): T {
  return JSON.parse(readFileSync(resolve(BUNDLE_DIR, rel), 'utf8')) as T
}

export function readBytes(rel: string): Uint8Array {
  return new Uint8Array(readFileSync(resolve(BUNDLE_DIR, rel)))
}
