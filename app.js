// Frontend-only vector-space explorer.
// Embeds text via Transformers.js in the browser, projects to 2D or 3D with PCA,
// renders a scatter plot, and lets the user drop in new points.

import { getExtractor, embed } from './embed.js';
import { pcaReduce, pcaProject } from './pca.js';

// ---------- DOM helpers ----------
const el = (id) => document.getElementById(id);
const $model = el('model');
const $maxlen = el('maxlen');
const $status = el('status');
const $batchText = el('batch-text');
const $addBatch = el('add-batch');
const $clearDataset = el('clear-dataset');
const $loadExample = el('load-example');
const $fileInput = el('file-input');
const $datasetSummary = el('dataset-summary');
const $newText = el('new-text');
const $addNew = el('add-new');
const $newList = el('new-list');
const $canvas = el('chart');
const $threeContainer = el('three-container');
const $tooltip = el('tooltip');
const $legend = el('legend');
const $variance = el('variance');
const $segBtns = document.querySelectorAll('.seg-btn');

// Read the current "Context" setting as a number or null (auto = no cap).
function currentMaxLength() {
  const v = $maxlen.value;
  if (v === 'auto') return null;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

// ---------- state ----------
const COLORS = {
  dataset: '#7aa2ff',     // single, consistent color for the whole dataset
  datasetEdge: '#0f1115', // outline so light-blue points read on the dark canvas
  newPoint: '#ff6b9b',    // contrasting pink so new points are unmistakable
  axis: '#3a4257',
  grid: '#252b3a',
};

const dataset = {
  items: [],       // { id, label }
  vectors: null,   // Float32Array with ._n and ._d
  pca: null,       // pcaReduce result
  extractor: null,
  modelId: null,
};

const newPoints = []; // { id, label }

let currentDim = 2; // 2 or 3

// Built-in sample so users can see the tool work without typing anything.
const EXAMPLE_DATASET = [
  'a photo of a red rose',
  'a bouquet of tulips',
  'a sunflower in a field',
  'a bowl of fresh pasta',
  'a slice of pepperoni pizza',
  'a cup of black coffee',
  'a glass of orange juice',
  'a thunderstorm in the mountains',
  'rain pattering on a window',
  'ocean waves crashing on rocks',
  'a violin playing a soft melody',
  'a piano concerto in a concert hall',
  'a guitar solo around a campfire',
  'a tabby cat sleeping on a couch',
  'a golden retriever playing fetch',
  'a hummingbird hovering at a flower',
  'a basketball game in a stadium',
  'a tennis match on a clay court',
  'a chess match in a quiet park',
  'a marathon runner crossing the finish line',
  'a programmer debugging a tricky bug',
  'an artist painting a sunset',
  'a chef plating a gourmet dish',
  'a scientist peering into a microscope',
  'a teacher writing on a chalkboard',
  'a child building a sandcastle',
  'a toddler taking their first steps',
  'a teenager riding a skateboard',
  'an elderly couple walking hand in hand',
  'a baby laughing at a peek-a-boo',
];

// ---------- status helpers ----------
function setStatus(msg, cls = '') {
  $status.textContent = msg;
  $status.className = 'status ' + cls;
}

function cryptoRandomId() {
  return Math.random().toString(36).slice(2, 10);
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

// ---------- extractors ----------
let activeModelPromise = null;
async function loadExtractor(modelId) {
  if (dataset.extractor && dataset.modelId === modelId) return dataset.extractor;
  if (activeModelPromise) return activeModelPromise;
  setStatus(`Loading model ${modelId}...`, 'busy');
  activeModelPromise = getExtractor(modelId).then((e) => {
    dataset.extractor = e;
    dataset.modelId = modelId;
    setStatus(`Model ${modelId} loaded.`, 'ok');
    return e;
  }).catch((err) => {
    setStatus(`Failed to load model: ${err.message}`, 'error');
    activeModelPromise = null;
    throw err;
  });
  return activeModelPromise;
}

// ---------- embedding helpers ----------
function parseLines(text) {
  return text.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
}

async function embedTexts(extractor, texts) {
  if (!texts.length) return null;
  const max_length = currentMaxLength();
  return await embed(extractor, texts, { max_length });
}

async function refreshDataset() {
  if (!dataset.items.length) {
    dataset.vectors = null;
    dataset.pca = null;
    draw();
    $datasetSummary.textContent = '0 items';
    return;
  }
  $datasetSummary.textContent = `${dataset.items.length} items`;
  const extractor = await loadExtractor($model.value);
  setStatus(`Embedding ${dataset.items.length} items...`, 'busy');
  const labels = dataset.items.map((i) => i.label);
  const vectors = await embedTexts(extractor, labels);
  if (!vectors) return;
  dataset.vectors = vectors;
  const k = currentDim;
  setStatus(`Running ${k}D PCA on ${dataset.items.length} × ${vectors._d} vectors...`, 'busy');
  const pca = pcaReduce(vectors, k);
  dataset.pca = pca;
  for (let i = 0; i < dataset.items.length; i++) {
    dataset.items[i].x = pca.points[i * k];
    dataset.items[i].y = pca.points[i * k + 1];
    dataset.items[i].z = k === 3 ? pca.points[i * k + 2] : 0;
  }
  // Re-project any existing new points so they stay in sync
  await reprojectNewPoints();
  draw();
  const ve = pca.varianceExplained;
  const total = (ve[0] + ve[1]) * 100;
  const zStr = ve[2] !== undefined ? ` · PC3 ${(ve[2] * 100).toFixed(1)}%` : '';
  setStatus(`Done. PC1 ${(ve[0] * 100).toFixed(1)}%, PC2 ${(ve[1] * 100).toFixed(1)}%${zStr} (total ${total.toFixed(1)}%).`, 'ok');
  $variance.textContent = `PC1 ${(ve[0] * 100).toFixed(2)}% · PC2 ${(ve[1] * 100).toFixed(2)}%${zStr ? ' · PC3 ' + (ve[2] * 100).toFixed(2) + '%' : ''} · total ${total.toFixed(2)}%`;
}

async function reprojectNewPoints() {
  if (!newPoints.length || !dataset.pca || !dataset.extractor) return;
  const labels = newPoints.map((p) => p.label);
  const vectors = await embedTexts(dataset.extractor, labels);
  if (!vectors) return;
  const k = dataset.pca.rank;
  const projected = pcaProject(vectors, dataset.pca);
  for (let i = 0; i < newPoints.length; i++) {
    newPoints[i].x = projected[i * k];
    newPoints[i].y = projected[i * k + 1];
    newPoints[i].z = k === 3 ? projected[i * k + 2] : 0;
  }
}

async function addBatch() {
  const lines = parseLines($batchText.value);
  if (!lines.length) {
    setStatus('Nothing to add.', 'error');
    return;
  }
  for (const label of lines) {
    dataset.items.push({ id: cryptoRandomId(), label });
  }
  $batchText.value = '';
  await refreshDataset();
}

async function addNewPoint() {
  const label = $newText.value.trim();
  if (!label) {
    setStatus('Enter a description for the new point.', 'error');
    return;
  }
  if (!dataset.pca) {
    setStatus('Add a dataset first to compute PCA.', 'error');
    return;
  }
  // If a pre-embed for this exact text already completed, use it. Otherwise
  // do the embed now (this is the cold path that fires when the user types
  // very fast and clicks add before the debounce resolves).
  let v = preEmbedResult && preEmbedResult.text === label ? preEmbedResult.vectors : null;
  preEmbedResult = null;
  if (!v) {
    setStatus(`Embedding new point...`, 'busy');
    const extractor = await loadExtractor($model.value);
    v = await embedTexts(extractor, [label]);
    if (!v) return;
  }
  const projected = pcaProject(v, dataset.pca);
  const k = dataset.pca.rank;
  const point = {
    id: cryptoRandomId(),
    label,
    x: projected[0],
    y: projected[1],
    z: k === 3 ? projected[2] : 0,
  };
  newPoints.push(point);
  $newText.value = '';
  renderNewChips();
  draw();
  setStatus(`New point placed at (${point.x.toFixed(2)}, ${point.y.toFixed(2)}${k === 3 ? ', ' + point.z.toFixed(2) : ''}).`, 'ok');
}

function clearDataset() {
  dataset.items = [];
  dataset.vectors = null;
  dataset.pca = null;
  newPoints.length = 0;
  renderNewChips();
  $datasetSummary.textContent = '0 items';
  draw();
  setStatus('Dataset cleared.', '');
}

function removeNewPoint(id) {
  const idx = newPoints.findIndex((p) => p.id === id);
  if (idx >= 0) newPoints.splice(idx, 1);
  renderNewChips();
  draw();
}

function renderNewChips() {
  $newList.innerHTML = '';
  for (const p of newPoints) {
    const chip = document.createElement('span');
    chip.className = 'chip new';
    chip.textContent = truncate(p.label, 28);
    const x = document.createElement('span');
    x.className = 'x';
    x.textContent = '×';
    x.title = 'Remove';
    x.onclick = () => removeNewPoint(p.id);
    chip.appendChild(x);
    $newList.appendChild(chip);
  }
}

async function loadExample() {
  dataset.items = EXAMPLE_DATASET.map((label) => ({ id: cryptoRandomId(), label }));
  $batchText.value = '';
  await refreshDataset();
}

// ---------- 2D canvas drawing ----------
const ctx = $canvas.getContext('2d');
function setupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = $canvas.clientWidth;
  const h = $canvas.clientHeight || Math.round(w * 0.55);
  $canvas.width = Math.round(w * dpr);
  $canvas.height = Math.round(h * dpr);
  $canvas.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

function getBounds2D() {
  const pts = [];
  for (const i of dataset.items) pts.push([i.x, i.y]);
  for (const p of newPoints) pts.push([p.x, p.y]);
  if (!pts.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const padX = (maxX - minX) * 0.08 || 1;
  const padY = (maxY - minY) * 0.08 || 1;
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
}

function draw2D() {
  const { w, h } = setupCanvas();
  ctx.clearRect(0, 0, w, h);

  const b = getBounds2D();
  const sx = (v) => ((v - b.minX) / (b.maxX - b.minX)) * w;
  const sy = (v) => h - ((v - b.minY) / (b.maxY - b.minY)) * h;

  // grid
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i++) {
    const x = (i / 10) * w;
    const y = (i / 10) * h;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  ctx.strokeStyle = COLORS.axis;
  if (b.minX < 0 && b.maxX > 0) {
    const x0 = sx(0);
    ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, h); ctx.stroke();
  }
  if (b.minY < 0 && b.maxY > 0) {
    const y0 = sy(0);
    ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke();
  }

  // dataset: all same color
  ctx.fillStyle = COLORS.dataset;
  for (const it of dataset.items) {
    const px = sx(it.x);
    const py = sy(it.y);
    ctx.beginPath(); ctx.arc(px, py, 5.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = COLORS.datasetEdge;
  ctx.lineWidth = 1.5;
  for (const it of dataset.items) {
    const px = sx(it.x);
    const py = sy(it.y);
    ctx.beginPath(); ctx.arc(px, py, 5.5, 0, Math.PI * 2);
    ctx.stroke();
  }

  // new points: distinct color with glow
  for (const p of newPoints) {
    const px = sx(p.x);
    const py = sy(p.y);
    ctx.save();
    ctx.globalAlpha = 0.25;
    ctx.fillStyle = COLORS.newPoint;
    ctx.beginPath(); ctx.arc(px, py, 12, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    ctx.fillStyle = COLORS.newPoint;
    ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // store positions for tooltip hit-testing
  $canvas._positions = [];
  for (const it of dataset.items) $canvas._positions.push({ x: sx(it.x), y: sy(it.y), label: it.label, kind: 'dataset' });
  for (const p of newPoints) $canvas._positions.push({ x: sx(p.x), y: sy(p.y), label: p.label, kind: 'new' });
}

// ---------- 3D rendering (three.js) ----------
let threeState = null; // { scene, camera, renderer, controls, dataset, newPoints, raf }

async function ensureThree() {
  if (threeState) return threeState;
  // Dynamic import so the 2D path doesn't pay the three.js cost.
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');

  const scene = new THREE.Scene();
  scene.background = null;

  const container = $threeContainer;
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 500;
  const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 1000);
  camera.position.set(2.4, 2.0, 2.4);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(w, h);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  // axes
  const axes = new THREE.AxesHelper(1.2);
  scene.add(axes);

  // grid plane
  const grid = new THREE.GridHelper(2.4, 12, 0x2a3142, 0x1d2230);
  scene.add(grid);

  // small on-screen hint
  let hint = container.querySelector('.three-hint');
  if (!hint) {
    hint = document.createElement('div');
    hint.className = 'three-hint';
    hint.textContent = 'drag to rotate · scroll to zoom · right-drag to pan';
    container.appendChild(hint);
  }

  const pointsGroup = new THREE.Group();
  scene.add(pointsGroup);

  const onResize = () => {
    const nw = container.clientWidth;
    const nh = container.clientHeight;
    if (!nw || !nh) return;
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
    renderer.setSize(nw, nh);
  };
  window.addEventListener('resize', onResize);

  let raf = null;
  const animate = () => {
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(animate);
  };
  animate();

  threeState = { THREE, scene, camera, renderer, controls, pointsGroup, hint };
  return threeState;
}

function getBounds3D() {
  const pts = [];
  for (const i of dataset.items) pts.push([i.x, i.y, i.z]);
  for (const p of newPoints) pts.push([p.x, p.y, p.z]);
  if (!pts.length) return { cx: 0, cy: 0, cz: 0, range: 1 };
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const [x, y, z] of pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const range = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
  return { cx, cy, cz, range };
}

function draw3D() {
  ensureThree().then((t) => {
    const { THREE, pointsGroup, camera, controls } = t;
    // Clear previous points
    while (pointsGroup.children.length) {
      const c = pointsGroup.children.pop();
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    }

    const b = getBounds3D();
    // Re-center and scale: fit everything into roughly [-1, 1]
    const norm = (v) => v / b.range;
    const cx = b.cx, cy = b.cy, cz = b.cz;

    // dataset: single color
    const datasetGeom = new THREE.BufferGeometry();
    const dsPositions = new Float32Array(dataset.items.length * 3);
    for (let i = 0; i < dataset.items.length; i++) {
      dsPositions[i * 3]     = norm(dataset.items[i].x - cx);
      dsPositions[i * 3 + 1] = norm(dataset.items[i].y - cy);
      dsPositions[i * 3 + 2] = norm(dataset.items[i].z - cz);
    }
    datasetGeom.setAttribute('position', new THREE.BufferAttribute(dsPositions, 3));
    const dsMat = new THREE.PointsMaterial({ color: COLORS.dataset, size: 0.05, sizeAttenuation: true });
    pointsGroup.add(new THREE.Points(datasetGeom, dsMat));

    // new points: distinct color, slightly larger
    if (newPoints.length) {
      const newGeom = new THREE.BufferGeometry();
      const newPos = new Float32Array(newPoints.length * 3);
      for (let i = 0; i < newPoints.length; i++) {
        newPos[i * 3]     = norm(newPoints[i].x - cx);
        newPos[i * 3 + 1] = norm(newPoints[i].y - cy);
        newPos[i * 3 + 2] = norm(newPoints[i].z - cz);
      }
      newGeom.setAttribute('position', new THREE.BufferAttribute(newPos, 3));
      const newMat = new THREE.PointsMaterial({ color: COLORS.newPoint, size: 0.08, sizeAttenuation: true });
      pointsGroup.add(new THREE.Points(newGeom, newMat));
    }

    // Re-frame camera
    const target = new THREE.Vector3(0, 0, 0);
    const desiredDist = b.range * 2.4;
    camera.position.set(desiredDist, desiredDist * 0.8, desiredDist);
    camera.lookAt(target);
    controls.target.copy(target);
    controls.update();
  });
}

function draw() {
  if (currentDim === 2) draw2D();
  else draw3D();
  renderLegend();
}

function renderLegend() {
  $legend.innerHTML = '';
  const dsItem = document.createElement('span');
  dsItem.innerHTML = `<span class="swatch" style="background:${COLORS.dataset}"></span>dataset (${dataset.items.length})`;
  $legend.appendChild(dsItem);
  if (newPoints.length) {
    const nwItem = document.createElement('span');
    nwItem.innerHTML = `<span class="swatch" style="background:${COLORS.newPoint}"></span>new (${newPoints.length})`;
    $legend.appendChild(nwItem);
  }
}

// ---------- 2D tooltip on hover ----------
$canvas.addEventListener('mousemove', (e) => {
  if (currentDim !== 2) return;
  const rect = $canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const positions = $canvas._positions || [];
  let best = null, bestDist = 16;
  for (const p of positions) {
    const dx = mx - p.x, dy = my - p.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < bestDist) { bestDist = d; best = p; }
  }
  if (best) {
    $tooltip.hidden = false;
    $tooltip.textContent = (best.kind === 'new' ? '★ ' : '') + best.label;
    $tooltip.style.left = (mx + 12) + 'px';
    $tooltip.style.top = (my + 12) + 'px';
  } else {
    $tooltip.hidden = true;
  }
});
$canvas.addEventListener('mouseleave', () => { $tooltip.hidden = true; });

// ---------- file input ----------
async function readTextFile(file) {
  return await file.text();
}

$fileInput.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;
  setStatus(`Reading ${files.length} file(s)...`, 'busy');
  const all = [];
  for (const f of files) {
    const text = await readTextFile(f);
    for (const line of text.split(/\r?\n/)) {
      const t = line.trim();
      if (t) all.push(t);
    }
  }
  if (!all.length) { setStatus('No text found in files.', 'error'); return; }
  for (const label of all) {
    dataset.items.push({ id: cryptoRandomId(), label });
  }
  $fileInput.value = '';
  await refreshDataset();
});

// ---------- 2D / 3D toggle ----------
$segBtns.forEach((btn) => {
  btn.addEventListener('click', async () => {
    const dim = parseInt(btn.dataset.dim, 10);
    if (dim === currentDim) return;
    currentDim = dim;
    $segBtns.forEach((b) => b.classList.toggle('active', b === btn));
    if (dim === 2) {
      $threeContainer.hidden = true;
      $threeContainer.style.display = '';
      $canvas.hidden = false;
    } else {
      $canvas.hidden = true;
      $threeContainer.hidden = false;
      $threeContainer.style.display = 'block';
    }
    // Re-run PCA at the new dimensionality, then redraw.
    if (dataset.items.length) {
      await refreshDataset();
    } else {
      draw();
    }
  });
});

// ---------- event wiring ----------
$addBatch.addEventListener('click', () => addBatch());
$addNew.addEventListener('click', () => addNewPoint());
$clearDataset.addEventListener('click', () => clearDataset());
$loadExample.addEventListener('click', () => loadExample());
$model.addEventListener('change', async () => {
  dataset.extractor = null; // force reload
  await refreshDataset();
});
$maxlen.addEventListener('change', async () => {
  // Tokenization / embeddings depend on the max-length cap. Re-embed.
  if (dataset.items.length) await refreshDataset();
});

$batchText.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    addBatch();
  }
});

// Debounced pre-embed of the new-point text. While the user is typing, we
// run the model in the background so the eventual "Add new point" is instant.
// Cancellable: every keystroke aborts the previous in-flight embed.
const NEW_PREEMBED_DELAY_MS = 350;
let preEmbedAbort = null;
let preEmbedToken = 0;
let preEmbedInFlight = null; // latest Promise from embedTexts for the pending text
let preEmbedResult = null;   // { text, vectors } once the latest in-flight resolves
$newText.addEventListener('input', () => {
  if (preEmbedAbort) preEmbedAbort.abort = true;
  const myToken = ++preEmbedToken;
  const text = $newText.value.trim();
  if (!text || !dataset.extractor) {
    preEmbedResult = null;
    return;
  }
  const aborter = { abort: false };
  preEmbedAbort = aborter;
  setTimeout(async () => {
    if (aborter.abort || myToken !== preEmbedToken) return;
    try {
      const vectors = await embedTexts(dataset.extractor, [text]);
      if (aborter.abort || myToken !== preEmbedToken) return;
      preEmbedResult = { text, vectors };
    } catch (e) {
      // best-effort: ignore failures; the explicit add still does the embed
    }
  }, NEW_PREEMBED_DELAY_MS);
});
$newText.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    addNewPoint();
  }
});

// initial draw
draw();
setStatus('Ready. Add some data to begin — or click "Try example".', '');
