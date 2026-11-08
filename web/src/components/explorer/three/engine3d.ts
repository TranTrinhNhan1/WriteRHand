/**
 * three.js side of the 3D network views. Loaded with a dynamic import (own chunk), so three.js never
 * reaches the main bundle. Renders on demand only: after a change, while the camera is still easing,
 * and never while the host is off-screen or the tab is hidden.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import { nodeGrid, type Layer3D, type MapLayer3D, type NetSpec, type NodeLayer3D } from './spec'

export interface Colors3D {
  bg: string
  ink: string
  ink2: string
  ink3: string
  line: string
  sunken: string
  surface: string
  pos: string
  neg: string
  paper: string
  padInk: string
  mono: string
}

export interface Pick3D {
  layer: number
  /** neuron (node layers), channel (map stacks with several channels) or pixel (single-channel maps) */
  index: number
}

export interface Net3DHandle {
  setRevealed(ids: ReadonlySet<string> | null): void
  setSelected(sel: Pick3D | null): void
  setActive(active: boolean): void
  resize(width: number, height: number): void
  /** stage height (px) at which the default view of this network fills a stage of the given width */
  preferredHeight(width: number): number
  rotate(dAzimuth: number, dPolar: number): void
  zoom(factor: number): void
  reset(): void
  pick(clientX: number, clientY: number): Pick3D | null
  dispose(): void
}

const NODE = 0.14
const GAP = 0.25
const MAP = 5.2 // world size of a 28-pixel map side
const SLAB = 0.2 // distance between the channel planes of a stack
const LAYER_GAP = 3.4
const THICK = 36 // strongest connections drawn as thick lines, per pair of layers
const MARGIN = 0.92 // the default view fills 92 % of the stage (4 % margin on each side)

/** World size of a map side: grows with its pixel count, compressed so a 5 × 5 map stays visible (labels give the true shape). */
const mapSide = (px: number) => MAP * Math.pow(px / 28, 0.6)

interface Placed {
  layer: Layer3D
  x0: number
  x1: number
  height: number
  width: number
  /** world position of item i (node index / pixel index of a single-channel map) */
  pos: (i: number) => THREE.Vector3 | null
}

const col = (s: string) => new THREE.Color(s || '#888888')

/** Mix in sRGB (as the 2D heat maps do), returned as a linear colour like every THREE.Color. */
function mix(a: THREE.Color, b: THREE.Color, t: number) {
  const x = a.clone().convertLinearToSRGB()
  return x.lerp(b.clone().convertLinearToSRGB(), Math.max(0, Math.min(1, t))).convertSRGBToLinear()
}

interface Palette {
  base: THREE.Color
  pos: THREE.Color
  neg: THREE.Color
  paper: THREE.Color
  padInk: THREE.Color
}

/** The heat palette of the 2D views: lime positive, orange negative, ink gray for images. */
function scaleColor(v: number, max: number, scale: 'signed' | 'positive' | 'gray', c: Palette, floor = 0, gamma = 1) {
  if (scale === 'gray') return mix(c.paper, c.padInk, v)
  const t = Math.pow(Math.min(1, Math.abs(v) / (max || 1)), gamma)
  if (v < 0 && scale === 'signed') return mix(c.base, c.neg, floor + (1 - floor) * t)
  if (v <= 0) return c.base.clone()
  return mix(c.base, c.pos, floor + (1 - floor) * t)
}

function absMax(v: ArrayLike<number>) {
  let m = 0
  for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]))
  return m || 1
}

/** 99th percentile of the non-zero |values|: one very strong cell does not leave every other map black. */
function robustMax(v: ArrayLike<number>) {
  const a: number[] = []
  for (let i = 0; i < v.length; i++) if (v[i] !== 0) a.push(Math.abs(v[i]))
  if (!a.length) return 1
  a.sort((x, y) => x - y)
  return a[Math.min(a.length - 1, Math.floor(a.length * 0.99))] || absMax(v)
}

export function createNet3D(host: HTMLElement, spec: NetSpec, colors: Colors3D, opts: { reducedMotion: boolean; onChange?: () => void }): Net3DHandle {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power', preserveDrawingBuffer: true })
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.setClearColor(col(colors.bg), 1)
  host.appendChild(renderer.domElement)
  renderer.domElement.style.display = 'block'
  renderer.domElement.style.width = '100%'
  renderer.domElement.style.height = '100%'
  // text is HTML laid over the canvas: crisp and readable at any zoom
  const overlay = document.createElement('div')
  overlay.className = 'ex-3d__labels'
  overlay.setAttribute('aria-hidden', 'true')
  host.appendChild(overlay)

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(22, 2, 0.1, 2000)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = !opts.reducedMotion
  controls.dampingFactor = 0.12
  controls.autoRotate = false
  controls.enablePan = true
  controls.screenSpacePanning = true
  // the page scrolls with the wheel; zoom uses the buttons or keys
  controls.enableZoom = false

  const C: Palette = { base: col(colors.sunken), pos: col(colors.pos), neg: col(colors.neg), paper: col(colors.paper), padInk: col(colors.padInk) }
  const mapBase = mix(C.paper, col(colors.sunken), 0.55) // empty map cells: a step above the screen so the slabs read as slabs
  const lineCol = col(colors.line)
  const ink = col(colors.ink)

  // ------------------------------------------------------------------ layout along x
  const placed: Placed[] = []
  let x = 0
  for (const layer of spec.layers) {
    if (layer.kind === 'maps') {
      const depth = (layer.c - 1) * SLAB
      const x0 = x
      const W = mapSide(layer.w)
      const H = mapSide(layer.h)
      placed.push({
        layer,
        x0,
        x1: x0 + depth,
        height: H,
        width: W,
        pos: (i) => (layer.c === 1 && i < layer.h * layer.w ? new THREE.Vector3(x0, H / 2 - ((Math.floor(i / layer.w) + 0.5) * H) / layer.h, -W / 2 + (((i % layer.w) + 0.5) * W) / layer.w) : null),
      })
      x = x0 + depth + LAYER_GAP
    } else {
      const g = nodeGrid(layer.shown.length, !!layer.names)
      const where = new Map(layer.shown.map((n, k) => [n, k]))
      const H = (g.rows - 1) * GAP
      const W = (g.cols - 1) * GAP
      const x0 = x
      placed.push({
        layer,
        x0,
        x1: x0,
        height: H,
        width: W,
        pos: (i) => {
          const k = where.get(i)
          if (k == null) return null
          return new THREE.Vector3(x0, H / 2 - Math.floor(k / g.cols) * GAP, -W / 2 + (k % g.cols) * GAP)
        },
      })
      x = x0 + LAYER_GAP * 1.25
    }
  }

  // ------------------------------------------------------------------ labels (HTML)
  const labels: { el: HTMLElement; at: THREE.Vector3; below?: THREE.Vector3; layer: boolean }[] = []
  const addLabel = (text: string, at: THREE.Vector3, cls: string, below?: THREE.Vector3) => {
    const el = document.createElement('span')
    el.className = cls
    el.textContent = text
    overlay.appendChild(el)
    labels.push({ el, at, below, layer: cls === 'ex-3d__label' })
  }

  // ------------------------------------------------------------------ meshes
  const disposables: { dispose: () => void }[] = []
  const pickables: THREE.Object3D[] = []
  interface Revealable {
    ids: string[]
    show: (on: boolean) => void
  }
  const revealables: Revealable[] = []
  const outlines: { layer: number; index: number | null; obj: THREE.Object3D }[] = []
  const thickMats: LineMaterial[] = []

  const blank = new THREE.MeshBasicMaterial({ color: mapBase, side: THREE.DoubleSide })
  disposables.push(blank)
  const frameMat = new THREE.LineBasicMaterial({ color: lineCol, transparent: true, opacity: 0.9 })
  disposables.push(frameMat)

  placed.forEach((p, li) => {
    // on a narrow stage every other layer name goes under its layer, so neighbours do not collide
    addLabel(p.layer.label, new THREE.Vector3((p.x0 + p.x1) / 2, p.height / 2 + 0.55, 0), 'ex-3d__label', li % 2 ? new THREE.Vector3((p.x0 + p.x1) / 2, -p.height / 2 - 0.55, 0) : undefined)
    if (p.layer.kind === 'maps') addMaps(p.layer as MapLayer3D, p, li)
    else addNodes(p.layer as NodeLayer3D, p, li)
  })

  function addMaps(layer: MapLayer3D, p: Placed, li: number) {
    const geo = new THREE.PlaneGeometry(p.width, p.height)
    geo.rotateY(-Math.PI / 2)
    disposables.push(geo)
    const frameGeo = new THREE.EdgesGeometry(geo)
    disposables.push(frameGeo)
    const n = layer.h * layer.w
    const max = layer.scale === 'gray' ? 1 : robustMax(layer.data)
    const pal = { ...C, base: mapBase }
    const mats: THREE.MeshBasicMaterial[] = []
    for (let ch = 0; ch < layer.c; ch++) {
      const buf = new Uint8Array(n * 4)
      for (let r = 0; r < layer.h; r++)
        for (let q = 0; q < layer.w; q++) {
          const v = layer.data[ch * n + r * layer.w + q]
          const c = scaleColor(v, max, layer.scale, layer.scale === 'gray' ? C : pal, 0, layer.scale === 'positive' ? 0.7 : 1)
          // DataTexture rows start at the bottom
          const o = ((layer.h - 1 - r) * layer.w + q) * 4
          c.convertLinearToSRGB()
          buf[o] = Math.round(c.r * 255)
          buf[o + 1] = Math.round(c.g * 255)
          buf[o + 2] = Math.round(c.b * 255)
          buf[o + 3] = 255
        }
      const tex = new THREE.DataTexture(buf, layer.w, layer.h, THREE.RGBAFormat)
      tex.colorSpace = THREE.SRGBColorSpace
      tex.magFilter = THREE.NearestFilter
      tex.minFilter = THREE.NearestFilter
      tex.needsUpdate = true
      const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })
      disposables.push(tex, mat)
      mats.push(mat)
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(p.x0 + ch * SLAB, 0, 0)
      mesh.userData = { layer: li, channel: ch }
      scene.add(mesh)
      pickables.push(mesh)
      const frame = new THREE.LineSegments(frameGeo, frameMat)
      frame.position.copy(mesh.position)
      scene.add(frame)
      const sel = new THREE.LineSegments(frameGeo, new THREE.LineBasicMaterial({ color: ink }))
      disposables.push(sel.material as THREE.Material)
      sel.position.copy(mesh.position).add(new THREE.Vector3(-0.01, 0, 0))
      sel.visible = false
      sel.renderOrder = 5
      scene.add(sel)
      outlines.push({ layer: li, index: layer.c === 1 ? null : ch, obj: sel })
    }
    revealables.push({
      ids: layer.layerIds,
      show: (on) => {
        scene.traverse((o) => {
          if (o instanceof THREE.Mesh && o.userData.layer === li) o.material = on ? mats[o.userData.channel] : blank
        })
      },
    })
  }

  function addNodes(layer: NodeLayer3D, p: Placed, li: number) {
    const geo = new THREE.BoxGeometry(NODE, NODE, NODE)
    disposables.push(geo)
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff })
    disposables.push(mat)
    const mesh = new THREE.InstancedMesh(geo, mat, layer.shown.length)
    const m = new THREE.Matrix4()
    const max = absMax(Array.from(layer.shown, (i) => layer.values[i]))
    const on: THREE.Color[] = []
    layer.shown.forEach((i, k) => {
      const v = p.pos(i)!
      m.makeTranslation(v.x, v.y, v.z)
      mesh.setMatrixAt(k, m)
      const c = scaleColor(layer.values[i], max, layer.signed ? 'signed' : 'positive', C, 0.2, 0.8)
      on.push(c)
      mesh.setColorAt(k, c)
    })
    mesh.userData = { layer: li, shown: layer.shown }
    scene.add(mesh)
    pickables.push(mesh)
    revealables.push({
      ids: layer.layerIds,
      show: (vis) => {
        on.forEach((c, k) => mesh.setColorAt(k, vis ? c : C.base))
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
      },
    })
    const sel = new THREE.Mesh(new THREE.BoxGeometry(NODE * 2.4, NODE * 2.4, NODE * 2.4), new THREE.MeshBasicMaterial({ color: ink, wireframe: true }))
    disposables.push(sel.geometry, sel.material as THREE.Material)
    sel.visible = false
    scene.add(sel)
    outlines.push({ layer: li, index: -1, obj: sel })
    if (layer.names) {
      layer.shown.forEach((i) => {
        const name = layer.names![i] ?? String(i)
        const v = p.pos(i)!
        addLabel(name.length > 11 ? `${name.slice(0, 10)}…` : name, new THREE.Vector3(v.x, v.y, v.z + 0.22), 'ex-3d__cls')
      })
    }
    if (layer.total > layer.shown.length) addLabel(`+${layer.total - layer.shown.length}`, new THREE.Vector3(p.x0, -p.height / 2 - 0.5, 0), 'ex-3d__label')
  }

  // ------------------------------------------------------------------ connections
  for (const e of spec.edges) {
    const a = placed[e.from]
    const b = placed[e.to]
    const pos: number[] = []
    const rgba: number[] = []
    const tPos: number[] = []
    const tCol: number[] = []
    let max = 0
    for (const s of e.list) max = Math.max(max, Math.abs(s.v))
    e.list.forEach((s, k) => {
      const pa = a.pos(s.i)
      const pb = b.pos(s.j)
      if (!pa || !pb) return
      pa.x = a.x1
      const m = Math.abs(s.v) / (max || 1)
      const c = s.v >= 0 ? C.pos : C.neg
      if (k < THICK) {
        // the list is sorted by |a·w|: the first ones are the strongest
        tPos.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z)
        const cc = mix(C.paper, c, 0.45 + 0.55 * m)
        tCol.push(cc.r, cc.g, cc.b, cc.r, cc.g, cc.b)
      } else {
        pos.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z)
        const al = 0.14 + 0.7 * m
        rgba.push(c.r, c.g, c.b, al, c.r, c.g, c.b, al)
      }
    })
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(rgba, 4))
    const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false })
    disposables.push(geo, mat)
    const lines = new THREE.LineSegments(geo, mat)
    scene.add(lines)
    const tGeo = new LineSegmentsGeometry()
    if (tPos.length) {
      tGeo.setPositions(tPos)
      tGeo.setColors(tCol)
    }
    const tMat = new LineMaterial({ linewidth: 2.2, vertexColors: true, worldUnits: false, transparent: true, opacity: 0.95, depthWrite: false })
    thickMats.push(tMat)
    disposables.push(tGeo, tMat)
    const thick = new LineSegments2(tGeo, tMat)
    thick.visible = tPos.length > 0
    scene.add(thick)
    revealables.push({
      ids: [e.layerId],
      show: (on) => {
        lines.visible = on
        thick.visible = on && tPos.length > 0
      },
    })
  }

  // ------------------------------------------------------------------ camera: fit the drawn content
  const pts: THREE.Vector3[] = []
  for (const p of placed) {
    const hw = p.width / 2 + (p.layer.kind === 'nodes' && p.layer.names ? 1.3 : 0.1)
    const hh = p.height / 2
    for (const xx of [p.x0, p.x1]) for (const yy of [-hh - 0.9, hh + 0.9]) for (const zz of [-hw, hw]) pts.push(new THREE.Vector3(xx, yy, zz))
  }
  const box = new THREE.Box3().setFromPoints(pts)
  const center = box.getCenter(new THREE.Vector3())
  const radius = box.getBoundingSphere(new THREE.Sphere()).radius
  const span = radius * 2
  const wideDir = new THREE.Vector3(-0.5, 0.36, 0.79).normalize()
  // a narrow (phone) stage looks more along the layer axis, so the network recedes in depth instead of becoming a thin strip
  const narrowDir = new THREE.Vector3(-0.78, 0.34, 0.52).normalize()
  const dir = wideDir.clone()
  const ndc = new THREE.Vector3()

  /** projected bounds of the content (NDC) for the current camera */
  const bounds = () => {
    camera.updateMatrixWorld()
    let x0 = Infinity
    let x1 = -Infinity
    let y0 = Infinity
    let y1 = -Infinity
    for (const p of pts) {
      ndc.copy(p).project(camera)
      x0 = Math.min(x0, ndc.x)
      x1 = Math.max(x1, ndc.x)
      y0 = Math.min(y0, ndc.y)
      y1 = Math.max(y1, ndc.y)
    }
    return { x0, x1, y0, y1 }
  }

  /** Default view: looking along `dir`, centred and scaled so the content fills MARGIN of the stage in width or height. */
  const home = () => {
    dir.copy(camera.aspect < 1.5 ? narrowDir : wideDir)
    const target = center.clone()
    let dist = radius * 2.4
    const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dir).normalize()
    const up = new THREE.Vector3().crossVectors(dir, right).normalize()
    const tv = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)
    for (let it = 0; it < 6; it++) {
      camera.position.copy(target).addScaledVector(dir, dist)
      camera.lookAt(target)
      const b = bounds()
      const cx = (b.x0 + b.x1) / 2
      const cy = (b.y0 + b.y1) / 2
      target.addScaledVector(right, cx * dist * tv * camera.aspect).addScaledVector(up, cy * dist * tv)
      const s = Math.max((b.x1 - b.x0) / 2, (b.y1 - b.y0) / 2) / MARGIN
      dist *= s
    }
    camera.position.copy(target).addScaledVector(dir, dist)
    controls.target.copy(target)
    camera.lookAt(target)
    controls.update()
  }

  // ------------------------------------------------------------------ on-demand rendering
  let active = true
  let raf = 0
  let disposed = false
  let width = 1
  let height = 1
  const placeLabels = () => {
    camera.updateMatrixWorld()
    // a label that would overlap one already placed is hidden (layer names first, then class names)
    const taken: { x0: number; x1: number; y0: number; y1: number }[] = []
    const narrow = width < 560
    for (const l of labels) {
      const under = narrow && !!l.below
      l.el.classList.toggle('is-below', under)
      ndc.copy(under ? l.below! : l.at).project(camera)
      const px = ((ndc.x + 1) / 2) * width
      const py = ((1 - ndc.y) / 2) * height
      l.el.style.transform = `translate(${px}px, ${py}px)`
      const w = l.el.offsetWidth
      const h = l.el.offsetHeight
      const r = l.layer ? { x0: px - w / 2, x1: px + w / 2, y0: under ? py : py - h, y1: under ? py + h : py } : { x0: px + 4, x1: px + 4 + w, y0: py - h / 2, y1: py + h / 2 }
      const off = ndc.z > 1 || ndc.z < -1 || r.x0 < 0 || r.x1 > width || r.y0 < 0 || r.y1 > height
      const clash = taken.some((t) => r.x0 < t.x1 + 2 && r.x1 > t.x0 - 2 && r.y0 < t.y1 - 2 && r.y1 > t.y0 + 2)
      const show = !off && !clash
      l.el.style.visibility = show ? 'visible' : 'hidden'
      if (show) taken.push(r)
    }
  }
  const frame = () => {
    raf = 0
    if (disposed || !active) return
    const moving = controls.update()
    renderer.render(scene, camera)
    placeLabels()
    if (moving && controls.enableDamping) raf = requestAnimationFrame(frame)
  }
  const request = () => {
    // off-screen or hidden tab: nothing is drawn; setActive(true) draws the current state
    if (disposed || !active) return
    if (!raf) raf = requestAnimationFrame(frame)
  }
  let moved = false
  const onStart = () => (moved = true)
  controls.addEventListener('start', onStart)
  const onCtl = () => {
    request()
    opts.onChange?.()
  }
  controls.addEventListener('change', onCtl)

  const ray = new THREE.Raycaster()
  const spherical = new THREE.Spherical()

  return {
    setRevealed(ids) {
      for (const r of revealables) r.show(!ids || r.ids.some((id) => ids.has(id)))
      request()
    },
    setSelected(sel) {
      for (const o of outlines) {
        o.obj.visible = false
        if (!sel || o.layer !== sel.layer) continue
        if (o.index === -1) {
          const p = placed[sel.layer].pos(sel.index)
          if (p) {
            o.obj.position.copy(p)
            o.obj.visible = true
          }
        } else if (o.index === null || o.index === sel.index) o.obj.visible = true
      }
      request()
    },
    setActive(a) {
      active = a
      // the browser may have dropped the last frame while the canvas was hidden: draw it again
      if (a) request()
    },
    resize(w, h) {
      if (w <= 0 || h <= 0) return
      width = w
      height = h
      renderer.setSize(w, h, false)
      for (const m of thickMats) m.resolution.set(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      if (!moved) home()
      request()
    },
    preferredHeight(w) {
      // content aspect of the default view, measured with a wide camera
      const keep = camera.aspect
      camera.aspect = 2
      camera.updateProjectionMatrix()
      home()
      const b = bounds()
      const aspect = ((b.x1 - b.x0) * camera.aspect) / Math.max(1e-6, b.y1 - b.y0)
      camera.aspect = keep
      camera.updateProjectionMatrix()
      home()
      return w / aspect
    },
    rotate(dAz, dPol) {
      moved = true
      const off = camera.position.clone().sub(controls.target)
      spherical.setFromVector3(off)
      spherical.theta += dAz
      spherical.phi = Math.max(0.15, Math.min(Math.PI - 0.15, spherical.phi + dPol))
      off.setFromSpherical(spherical)
      camera.position.copy(controls.target).add(off)
      camera.lookAt(controls.target)
      controls.update()
      request()
    },
    zoom(f) {
      moved = true
      const off = camera.position.clone().sub(controls.target)
      const len = Math.max(span * 0.1, Math.min(span * 4, off.length() * f))
      camera.position.copy(controls.target).add(off.setLength(len))
      controls.update()
      request()
    },
    reset() {
      moved = false
      home()
      request()
    },
    pick(cx, cy) {
      const r = renderer.domElement.getBoundingClientRect()
      ray.setFromCamera(new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1), camera)
      const hit = ray.intersectObjects(pickables, false)[0]
      if (!hit) return null
      const ud = hit.object.userData as { layer: number; channel?: number; shown?: number[] }
      if (ud.shown && hit.instanceId != null) return { layer: ud.layer, index: ud.shown[hit.instanceId] }
      const layer = spec.layers[ud.layer]
      if (layer.kind === 'maps' && layer.c === 1 && hit.uv) {
        const q = Math.min(layer.w - 1, Math.floor(hit.uv.x * layer.w))
        const rr = Math.min(layer.h - 1, Math.floor((1 - hit.uv.y) * layer.h))
        return { layer: ud.layer, index: rr * layer.w + q }
      }
      return { layer: ud.layer, index: ud.channel ?? 0 }
    },
    dispose() {
      disposed = true
      if (raf) cancelAnimationFrame(raf)
      controls.removeEventListener('change', onCtl)
      controls.removeEventListener('start', onStart)
      controls.dispose()
      for (const d of disposables) d.dispose()
      scene.traverse((o) => {
        if (o instanceof THREE.InstancedMesh) o.dispose()
      })
      scene.clear()
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.remove()
      overlay.remove()
    },
  }
}

/** True when this browser can create a WebGL context (checked before three.js would log an error). */
export function webglAvailable(): boolean {
  try {
    const c = document.createElement('canvas')
    const gl = (c.getContext('webgl2') ?? c.getContext('webgl')) as WebGLRenderingContext | null
    const ok = !!gl
    ;(gl?.getExtension('WEBGL_lose_context') as { loseContext?: () => void } | null)?.loseContext?.()
    return ok
  } catch {
    return false
  }
}
