// PCA via power iteration with deflation. Computes top-k eigvecs/eigvals only.
// When n < d, computes on the n x n covariance and lifts eigenvectors back to d-space.
// Shared between browser and Node.js.
// Input: number[][] (n rows, d cols) or Float32Array with ._n and ._d
// Output: { points: Float32Array(n*k) row-major, components: Float32Array(k*d) row-major,
//          mean: Float32Array(d), varianceExplained: number[k], rank, n, d, timingMs }
function now() { return (typeof performance !== 'undefined' ? performance.now() : Date.now()); }

function readInput(vectors) {
  let n, d, X;
  if (Array.isArray(vectors) && Array.isArray(vectors[0])) {
    n = vectors.length;
    d = vectors[0].length;
    X = new Float64Array(n * d);
    for (let i = 0; i < n; i++) {
      const row = vectors[i];
      for (let j = 0; j < d; j++) X[i * d + j] = row[j] ?? 0;
    }
  } else if (vectors instanceof Float32Array || vectors instanceof Float64Array) {
    if (!vectors._n || !vectors._d) {
      // Try to infer dimensions from a non-metadata case
      throw new Error('pcaReduce: TypedArray needs ._n and ._d metadata (use embed() from embed.js)');
    }
    n = vectors._n;
    d = vectors._d;
    X = vectors instanceof Float64Array ? new Float64Array(vectors) : Float64Array.from(vectors);
  } else {
    throw new Error('pcaReduce: unsupported input');
  }
  return { n, d, X };
}

// Power iteration for top-k eigvecs/eigvals of a symmetric d x d matrix.
// Returns {eigvals[k], V[d*k row-major, eigenvectors as columns sorted DESC]}.
// Uses deflation (Hotelling's) and re-orthogonalization for stability.
function powerIterationTopK(A0, d, k) {
  // Copy A so we can deflate without mutating the original
  const A = new Float64Array(A0);
  const eigvals = new Float64Array(k);
  const V = new Float64Array(d * k);

  const maxIter = 40;
  const tol = 1e-8;
  const v = new Float64Array(d);
  const y = new Float64Array(d);

  for (let j = 0; j < k; j++) {
    // Initialize v with deterministic seed based on column index to avoid
    // re-converging to previous eigenvectors
    const seed = (j * 0.63255813 + 0.12345) % 1;
    for (let i = 0; i < d; i++) v[i] = ((i + 1) * seed + (i * 0.211)) % 1;

    let prevLambda = 0;
    let lambda = 0;
    for (let iter = 0; iter < maxIter; iter++) {
      // y = A @ v
      for (let i = 0; i < d; i++) {
        let s = 0;
        const rowBase = i * d;
        for (let l = 0; l < d; l++) s += A[rowBase + l] * v[l];
        y[i] = s;
      }
      // lambda = v . y (Rayleigh quotient)
      lambda = 0;
      for (let i = 0; i < d; i++) lambda += v[i] * y[i];
      // normalize y
      let norm = 0;
      for (let i = 0; i < d; i++) norm += y[i] * y[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < d; i++) v[i] = y[i] / norm;
      if (iter > 5 && Math.abs(lambda - prevLambda) < tol * (Math.abs(lambda) + 1)) break;
      prevLambda = lambda;
    }

    // Orthogonalize v against previously found eigenvectors (Gram-Schmidt)
    for (let jp = 0; jp < j; jp++) {
      const vBase = jp * d;
      let dot = 0;
      for (let i = 0; i < d; i++) dot += v[i] * V[vBase + i];
      for (let i = 0; i < d; i++) v[i] -= dot * V[vBase + i];
    }
    let norm = 0;
    for (let i = 0; i < d; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < d; i++) v[i] /= norm;

    // Refine lambda with orthogonalized v
    for (let i = 0; i < d; i++) {
      let s = 0;
      const rowBase = i * d;
      for (let l = 0; l < d; l++) s += A[rowBase + l] * v[l];
      y[i] = s;
    }
    lambda = 0;
    for (let i = 0; i < d; i++) lambda += v[i] * y[i];

    // Save eigenvector as column j of V
    const vBase = j * d;
    for (let i = 0; i < d; i++) V[vBase + i] = v[i];
    eigvals[j] = lambda;

    // Deflate: A = A - lambda * v v'
    for (let i = 0; i < d; i++) {
      const vi = v[i];
      for (let l = 0; l < d; l++) {
        A[i * d + l] -= lambda * vi * v[l];
      }
    }
  }
  return { eigvals, V };
}

export function pcaReduce(vectors, k = 2) {
  const t0 = now();
  const { n, d, X } = readInput(vectors);
  if (n < 2) throw new Error('pcaReduce: need at least 2 rows');
  const tCenter = now();

  // Mean-center
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let j = 0; j < d; j++) mean[j] += X[base + j];
  }
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let j = 0; j < d; j++) X[base + j] -= mean[j];
  }
  const tCov = now();

  let eigvals, components, tJacobi;
  if (n <= d) {
    // Small covariance: n x n. C_lift = X X' / (n-1).
    const m = n;
    const Csmall = new Float64Array(m * m);
    // Unrolled 4x inner loop for V8 auto-vectorization
    const d4 = d & ~3;
    for (let i = 0; i < m; i++) {
      const xiBase = i * d;
      for (let j = i; j < m; j++) {
        const xjBase = j * d;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let l = 0; l < d4; l += 4) {
          s0 += X[xiBase + l]     * X[xjBase + l];
          s1 += X[xiBase + l + 1] * X[xjBase + l + 1];
          s2 += X[xiBase + l + 2] * X[xjBase + l + 2];
          s3 += X[xiBase + l + 3] * X[xjBase + l + 3];
        }
        let s = s0 + s1 + s2 + s3;
        for (let l = d4; l < d; l++) s += X[xiBase + l] * X[xjBase + l];
        s /= (n - 1);
        Csmall[i * m + j] = s;
        Csmall[j * m + i] = s;
      }
    }
    const rank = Math.min(k, m);
    const { eigvals: eSmall, V: Vsmall } = powerIterationTopK(Csmall, m, rank);
    tJacobi = now();
    // Map small eigenvectors to d-space: V_d[:, j] = X^T * U[:,j] / sigma_j
    // Where sigma_j = sqrt(eigval_j * (n-1)) (since eSmall[j] = sigma^2 / (n-1))
    components = new Float32Array(rank * d);
    for (let j = 0; j < rank; j++) {
      const sigma = Math.sqrt(Math.max(eSmall[j] * (n - 1), 0));
      if (sigma < 1e-12) continue;
      const invSigma = 1 / sigma;
      const cBase = j * d;
      const m4 = m & ~3;
      for (let l = 0; l < d; l++) {
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let r = 0; r < m4; r += 4) {
          s0 += X[r * d + l]     * Vsmall[j * m + r];
          s1 += X[(r+1) * d + l] * Vsmall[j * m + r + 1];
          s2 += X[(r+2) * d + l] * Vsmall[j * m + r + 2];
          s3 += X[(r+3) * d + l] * Vsmall[j * m + r + 3];
        }
        let s = s0 + s1 + s2 + s3;
        for (let r = m4; r < m; r++) s += X[r * d + l] * Vsmall[j * m + r];
        components[cBase + l] = s * invSigma;
      }
    }
    eigvals = eSmall;
  } else {
    // Large covariance: d x d
    const C = new Float64Array(d * d);
    const n4 = n & ~3;
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let r = 0; r < n4; r += 4) {
          s0 += X[r * d + i]         * X[r * d + j];
          s1 += X[(r + 1) * d + i]   * X[(r + 1) * d + j];
          s2 += X[(r + 2) * d + i]   * X[(r + 2) * d + j];
          s3 += X[(r + 3) * d + i]   * X[(r + 3) * d + j];
        }
        let s = s0 + s1 + s2 + s3;
        for (let r = n4; r < n; r++) s += X[r * d + i] * X[r * d + j];
        s /= (n - 1);
        C[i * d + j] = s;
        C[j * d + i] = s;
      }
    }
    const rank = Math.min(k, d);
    const { eigvals: eLarge, V: Vlarge } = powerIterationTopK(C, d, rank);
    tJacobi = now();
    components = new Float32Array(rank * d);
    for (let j = 0; j < rank; j++) {
      for (let i = 0; i < d; i++) components[j * d + i] = Vlarge[j * d + i];
    }
    eigvals = eLarge;
  }

  // Project onto top-rank components
  const rank = Math.min(k, n);
  const points = new Float32Array(n * rank);
  const d4 = d & ~3;
  for (let i = 0; i < n; i++) {
    const xBase = i * d, yBase = i * rank;
    for (let j = 0; j < rank; j++) {
      const cBase = j * d;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let l = 0; l < d4; l += 4) {
        s0 += X[xBase + l]     * components[cBase + l];
        s1 += X[xBase + l + 1] * components[cBase + l + 1];
        s2 += X[xBase + l + 2] * components[cBase + l + 2];
        s3 += X[xBase + l + 3] * components[cBase + l + 3];
      }
      let s = s0 + s1 + s2 + s3;
      for (let l = d4; l < d; l++) s += X[xBase + l] * components[cBase + l];
      points[yBase + j] = s;
    }
  }

  let totalVar = 0;
  // trace(X X') / (n-1) = sum_i sum_j X[i,j]^2 / (n-1) for centered X.
  let tr = 0;
  for (let i = 0; i < n * d; i++) tr += X[i] * X[i];
  totalVar = tr / (n - 1);
  if (totalVar === 0) totalVar = 1;
  const varianceExplained = new Array(rank);
  for (let j = 0; j < rank; j++) {
    varianceExplained[j] = Math.max(eigvals[j], 0) / totalVar;
  }

  const tEnd = now();
  return {
    points,
    components,
    mean: new Float32Array(mean),
    varianceExplained,
    rank,
    n,
    d,
    timingMs: {
      total: Math.round(tEnd - t0),
      center: Math.round(tCenter - t0),
      cov: Math.round(tCov - tCenter),
      jacobi: Math.round(tJacobi - tCov),
      project: Math.round(tEnd - tJacobi),
    },
    diagnostics: { negEigenvalues: Array.from(eigvals).filter(v => v < -1e-10).length, mode: n <= d ? 'small_cov' : 'large_cov' },
  };
}

export function pcaProject(newVectors, pca) {
  let m, d;
  let X;
  if (Array.isArray(newVectors) && Array.isArray(newVectors[0])) {
    m = newVectors.length;
    d = newVectors[0].length;
    X = new Float64Array(m * d);
    for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) X[i * d + j] = newVectors[i][j] ?? 0;
  } else if (newVectors instanceof Float32Array || newVectors instanceof Float64Array) {
    m = newVectors._m ?? newVectors.length / newVectors._d;
    d = newVectors._d;
    X = newVectors instanceof Float64Array ? new Float64Array(newVectors) : Float64Array.from(newVectors);
  } else {
    throw new Error('pcaProject: unsupported input');
  }
  if (d !== pca.d) throw new Error(`pcaProject: dim mismatch ${d} vs ${pca.d}`);

  const rank = pca.rank;
  const mean = pca.mean;
  const components = pca.components;
  const d4 = d & ~3;
  const points = new Float32Array(m * rank);
  for (let i = 0; i < m; i++) {
    const xBase = i * d;
    const yBase = i * rank;
    for (let j = 0; j < d4; j += 4) {
      X[xBase + j]     -= mean[j];
      X[xBase + j + 1] -= mean[j + 1];
      X[xBase + j + 2] -= mean[j + 2];
      X[xBase + j + 3] -= mean[j + 3];
    }
    for (let j = d4; j < d; j++) X[xBase + j] -= mean[j];
    for (let j = 0; j < rank; j++) {
      const cBase = j * d;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let l = 0; l < d4; l += 4) {
        s0 += X[xBase + l]     * components[cBase + l];
        s1 += X[xBase + l + 1] * components[cBase + l + 1];
        s2 += X[xBase + l + 2] * components[cBase + l + 2];
        s3 += X[xBase + l + 3] * components[cBase + l + 3];
      }
      let s = s0 + s1 + s2 + s3;
      for (let l = d4; l < d; l++) s += X[xBase + l] * components[cBase + l];
      points[yBase + j] = s;
    }
  }
  return points;
}