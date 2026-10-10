// Ensures the embedding model is downloaded to the transformers.js cache.
// Idempotent: exits 0 if already cached.
import { pipeline, env } from '@huggingface/transformers';
import fs from 'node:fs';

env.allowRemoteModels = true;
env.useBrowserCache = false;

const MODEL = 'Xenova/all-MiniLM-L6-v2';
const cachePath = `${env.cacheDir.replace(/\\/g, '/')}${MODEL}`.replace(env.cacheDir, env.cacheDir.replace(/\\/g, '/'));

function isCached() {
  const dir = `${env.cacheDir.replace(/\\/g, '/')}`;
  const target = `${dir}${MODEL.split('/')[0]}/${MODEL.split('/')[1]}`;
  return fs.existsSync(target) && fs.readdirSync(target).length > 0;
}

if (!isCached()) {
  console.error(`[ensure_model] downloading ${MODEL}...`);
  const extractor = await pipeline('feature-extraction', MODEL, { quantized: true });
  await extractor(['warmup'], { pooling: 'mean', normalize: true });
  console.error('[ensure_model] cached.');
} else {
  console.error('[ensure_model] already cached.');
}