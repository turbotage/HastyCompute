/**
 * Volume — flat Float32Array storage in T×E×Z×Y×X C-order.
 *
 * Slice conventions (all textures are row-major, row = slow axis):
 *   Axial    (fixed z):  tex width = X, tex height = Y   → data layout Y×X
 *   Sagittal (fixed x):  tex width = Y, tex height = Z   → data layout Z×Y
 *   Coronal  (fixed y):  tex width = X, tex height = Z   → data layout Z×X
 */
export class Volume {
    /** @param {number[]} shape [T, E, Z, Y, X] */
    constructor(shape, data) {
        this.shape = shape;
        const [T, E, Z, Y, X] = shape;
        this.data = data ?? new Float32Array(T * E * Z * Y * X);
    }

    get T() { return this.shape[0]; }
    get E() { return this.shape[1]; }
    get Z() { return this.shape[2]; }
    get Y() { return this.shape[3]; }
    get X() { return this.shape[4]; }

    /** Flat index into this.data */
    idx(t, e, z, y, x) {
        const [T, E, Z, Y, X] = this.shape;
        return ((((t * E + e) * Z + z) * Y + y) * X + x);
    }

    /**
     * Axial slice (fixed z).
     * Returns a view (zero-copy) into this.data: shape Y×X.
     */
    getAxialSlice(t, e, z) {
        const [, E, Z, Y, X] = this.shape;
        const base = this.idx(t, e, z, 0, 0);
        // Contiguous in memory — return a view, not a copy.
        return this.data.subarray(base, base + Y * X);
    }

    /**
     * Sagittal slice (fixed x).
     * Returns a new Float32Array: shape Z×Y (row = z, col = y).
     */
    getSagittalSlice(t, e, x) {
        const [, , Z, Y] = this.shape;
        const slice = new Float32Array(Z * Y);
        for (let z = 0; z < Z; z++) {
            for (let y = 0; y < Y; y++) {
                slice[z * Y + y] = this.data[this.idx(t, e, z, y, x)];
            }
        }
        return slice;
    }

    /**
     * Coronal slice (fixed y).
     * Returns a new Float32Array: shape Z×X (row = z, col = x).
     */
    getCoronalSlice(t, e, y) {
        const [, , Z, , X] = this.shape;
        const slice = new Float32Array(Z * X);
        for (let z = 0; z < Z; z++) {
            for (let x = 0; x < X; x++) {
                slice[z * X + x] = this.data[this.idx(t, e, z, y, x)];
            }
        }
        return slice;
    }

    /**
     * Extra (T×E) slice at a fixed spatial voxel (z, y, x).
     * Returns a new Float32Array: shape T×E (row = t, col = e).
     * This is what the 4th panel displays; only the crosshair moves when t/e change.
     */
    getExtraSlice(z, y, x) {
        const [T, E] = this.shape;
        const slice = new Float32Array(T * E);
        for (let t = 0; t < T; t++) {
            for (let e = 0; e < E; e++) {
                slice[t * E + e] = this.data[this.idx(t, e, z, y, x)];
            }
        }
        return slice;
    }

    /**
     * Build a simple brain-like phantom for testing.
     * All values are in [0, 1].
     */
    static createSynthetic(shape) {        const [T, E, Z, Y, X] = shape;
        const vol = new Volume(shape);

        for (let t = 0; t < T; t++) {
            for (let e = 0; e < E; e++) {
                // Per-echo / per-time-point scale to make them distinguishable.
                const tScale = 1.0 + 0.15 * t / Math.max(1, T - 1);
                const eScale = 1.0 + 0.08 * e / Math.max(1, E - 1);

                for (let z = 0; z < Z; z++) {
                    // Normalized coordinates in [-1, 1]
                    const nz = (z + 0.5) / Z * 2.0 - 1.0;

                    for (let y = 0; y < Y; y++) {
                        const ny = (y + 0.5) / Y * 2.0 - 1.0;

                        for (let x = 0; x < X; x++) {
                            const nx = (x + 0.5) / X * 2.0 - 1.0;

                            // ── Outer skull-like ellipsoid ───────────────────
                            const skull = nx * nx / 0.72 + ny * ny / 0.90 + nz * nz / 0.56;
                            let val = skull <= 1.0 ? 0.35 : 0.0;

                            // ── CSF / ventricle (lower intensity core) ───────
                            const vent = (nx - 0.04) * (nx - 0.04) / 0.05 +
                                          ny * ny / 0.08 +
                                          nz * nz / 0.04;
                            if (vent <= 1.0) val = 0.08;

                            // ── Two "gray matter" lobes ──────────────────────
                            const gm1 = (nx - 0.30) * (nx - 0.30) / 0.08 +
                                        (ny - 0.15) * (ny - 0.15) / 0.08 +
                                        nz * nz / 0.06;
                            if (gm1 <= 1.0) val = Math.max(val, 0.75);

                            const gm2 = (nx + 0.30) * (nx + 0.30) / 0.08 +
                                        (ny - 0.15) * (ny - 0.15) / 0.08 +
                                        nz * nz / 0.06;
                            if (gm2 <= 1.0) val = Math.max(val, 0.75);

                            // ── Brain stem ───────────────────────────────────
                            const stem = nx * nx / 0.02 +
                                         (ny + 0.30) * (ny + 0.30) / 0.02 +
                                         nz * nz / 0.08;
                            if (stem <= 1.0) val = Math.max(val, 0.55);

                            // ── T/E variation + tiny noise ───────────────────
                            val *= tScale * eScale;
                            val += 0.012 * (Math.random() - 0.5);

                            vol.data[vol.idx(t, e, z, y, x)] =
                                Math.max(0.0, Math.min(1.0, val));
                        }
                    }
                }
            }
        }
        return vol;
    }
}

// ─── LazyVolume ───────────────────────────────────────────────────────────────

/**
 * LazyVolume — same slice API as Volume but generates data on-the-fly.
 * No backing array is allocated.  Designed for large volumes where pre-
 * allocating the full array would exceed memory, or to simulate gRPC
 * streaming where each slice arrives individually on demand.
 *
 * Slice computation is optimised by hoisting loop-invariant subexpressions
 * out of the inner loops.
 */
export class LazyVolume {
    /** @param {number[]} shape [T, E, Z, Y, X] */
    constructor(shape) {
        this.shape = shape;
    }

    get T() { return this.shape[0]; }
    get E() { return this.shape[1]; }
    get Z() { return this.shape[2]; }
    get Y() { return this.shape[3]; }
    get X() { return this.shape[4]; }

    // ── Axial slice (fixed z) — width=X, height=Y ────────────────────────────

    getAxialSlice(t, e, z) {
        const [T, E, Z, Y, X] = this.shape;
        const scale = (1.0 + 0.15 * t / Math.max(1, T - 1)) *
                      (1.0 + 0.08 * e / Math.max(1, E - 1));
        const nz = (z + 0.5) / Z * 2.0 - 1.0;
        // Hoist z-only terms.
        const nz2sk = nz * nz / 0.56;
        const nz2vt = nz * nz / 0.04;
        const nz2gm = nz * nz / 0.06;
        const nz2st = nz * nz / 0.08;

        const slice = new Float32Array(Y * X);
        let i = 0;
        for (let y = 0; y < Y; y++) {
            const ny = (y + 0.5) / Y * 2.0 - 1.0;
            const ny2sk = ny * ny / 0.90;
            const ny2vt = ny * ny / 0.08;
            const nyd   = ny - 0.15;  const ny2gm = nyd * nyd / 0.08;
            const nys   = ny + 0.30;  const ny2st = nys * nys / 0.02;
            for (let x = 0; x < X; x++) {
                const nx = (x + 0.5) / X * 2.0 - 1.0;
                const skull = nx * nx / 0.72 + ny2sk + nz2sk;
                let v = skull <= 1.0 ? 0.35 : 0.0;
                const nxv = nx - 0.04; const vent = nxv * nxv / 0.05 + ny2vt + nz2vt;
                if (vent <= 1.0) v = 0.08;
                const nxl = nx - 0.30; if (nxl * nxl / 0.08 + ny2gm + nz2gm <= 1.0) { if (v < 0.75) v = 0.75; }
                const nxr = nx + 0.30; if (nxr * nxr / 0.08 + ny2gm + nz2gm <= 1.0) { if (v < 0.75) v = 0.75; }
                if (nx * nx / 0.02 + ny2st + nz2st <= 1.0) { if (v < 0.55) v = 0.55; }
                v *= scale;
                slice[i++] = v > 1.0 ? 1.0 : (v < 0.0 ? 0.0 : v);
            }
        }
        return slice;
    }

    // ── Sagittal slice (fixed x) — width=Y, height=Z ─────────────────────────

    getSagittalSlice(t, e, x) {
        const [T, E, Z, Y, X] = this.shape;
        const scale = (1.0 + 0.15 * t / Math.max(1, T - 1)) *
                      (1.0 + 0.08 * e / Math.max(1, E - 1));
        const nx = (x + 0.5) / X * 2.0 - 1.0;
        // Hoist x-only terms.
        const nx2sk = nx * nx / 0.72;
        const nxv   = nx - 0.04; const nx2vt = nxv * nxv / 0.05;
        const nxl   = nx - 0.30; const nx2gl = nxl * nxl / 0.08;
        const nxr   = nx + 0.30; const nx2gr = nxr * nxr / 0.08;
        const nx2st = nx * nx / 0.02;

        const slice = new Float32Array(Z * Y);
        let i = 0;
        for (let z = 0; z < Z; z++) {
            const nz = (z + 0.5) / Z * 2.0 - 1.0;
            const nz2sk = nz * nz / 0.56;
            const nz2vt = nz * nz / 0.04;
            const nz2gm = nz * nz / 0.06;
            const nz2st = nz * nz / 0.08;
            for (let y = 0; y < Y; y++) {
                const ny = (y + 0.5) / Y * 2.0 - 1.0;
                const skull = nx2sk + ny * ny / 0.90 + nz2sk;
                let v = skull <= 1.0 ? 0.35 : 0.0;
                const vent = nx2vt + ny * ny / 0.08 + nz2vt;
                if (vent <= 1.0) v = 0.08;
                const nyd = ny - 0.15; const ny2gm = nyd * nyd / 0.08;
                if (nx2gl + ny2gm + nz2gm <= 1.0) { if (v < 0.75) v = 0.75; }
                if (nx2gr + ny2gm + nz2gm <= 1.0) { if (v < 0.75) v = 0.75; }
                const nys = ny + 0.30;
                if (nx2st + nys * nys / 0.02 + nz2st <= 1.0) { if (v < 0.55) v = 0.55; }
                v *= scale;
                slice[i++] = v > 1.0 ? 1.0 : (v < 0.0 ? 0.0 : v);
            }
        }
        return slice;
    }

    // ── Coronal slice (fixed y) — width=X, height=Z ──────────────────────────

    getCoronalSlice(t, e, y) {
        const [T, E, Z, Y, X] = this.shape;
        const scale = (1.0 + 0.15 * t / Math.max(1, T - 1)) *
                      (1.0 + 0.08 * e / Math.max(1, E - 1));
        const ny = (y + 0.5) / Y * 2.0 - 1.0;
        // Hoist y-only terms.
        const ny2sk = ny * ny / 0.90;
        const ny2vt = ny * ny / 0.08;
        const nyd   = ny - 0.15; const ny2gm = nyd * nyd / 0.08;
        const nys   = ny + 0.30; const ny2st = nys * nys / 0.02;

        const slice = new Float32Array(Z * X);
        let i = 0;
        for (let z = 0; z < Z; z++) {
            const nz = (z + 0.5) / Z * 2.0 - 1.0;
            const nz2sk = nz * nz / 0.56;
            const nz2vt = nz * nz / 0.04;
            const nz2gm = nz * nz / 0.06;
            const nz2st = nz * nz / 0.08;
            for (let x = 0; x < X; x++) {
                const nx = (x + 0.5) / X * 2.0 - 1.0;
                const skull = nx * nx / 0.72 + ny2sk + nz2sk;
                let v = skull <= 1.0 ? 0.35 : 0.0;
                const nxv = nx - 0.04; const vent = nxv * nxv / 0.05 + ny2vt + nz2vt;
                if (vent <= 1.0) v = 0.08;
                const nxl = nx - 0.30; if (nxl * nxl / 0.08 + ny2gm + nz2gm <= 1.0) { if (v < 0.75) v = 0.75; }
                const nxr = nx + 0.30; if (nxr * nxr / 0.08 + ny2gm + nz2gm <= 1.0) { if (v < 0.75) v = 0.75; }
                if (nx * nx / 0.02 + ny2st + nz2st <= 1.0) { if (v < 0.55) v = 0.55; }
                v *= scale;
                slice[i++] = v > 1.0 ? 1.0 : (v < 0.0 ? 0.0 : v);
            }
        }
        return slice;
    }

    // ── Extra slice (fixed z,y,x) — width=E, height=T ────────────────────────

    getExtraSlice(z, y, x) {
        const [T, E, Z, Y, X] = this.shape;
        const nz = (z + 0.5) / Z * 2.0 - 1.0;
        const ny = (y + 0.5) / Y * 2.0 - 1.0;
        const nx = (x + 0.5) / X * 2.0 - 1.0;
        const nz2sk = nz * nz / 0.56; const nz2vt = nz * nz / 0.04;
        const nz2gm = nz * nz / 0.06; const nz2st = nz * nz / 0.08;
        const ny2sk = ny * ny / 0.90;  const ny2vt = ny * ny / 0.08;
        const nyd = ny - 0.15; const ny2gm = nyd * nyd / 0.08;
        const nys = ny + 0.30; const ny2st = nys * nys / 0.02;
        const skull = nx * nx / 0.72 + ny2sk + nz2sk;
        const nxv = nx - 0.04; const vent = nxv * nxv / 0.05 + ny2vt + nz2vt;
        const nxl = nx - 0.30; const gml = nxl * nxl / 0.08 + ny2gm + nz2gm;
        const nxr = nx + 0.30; const gmr = nxr * nxr / 0.08 + ny2gm + nz2gm;
        const nxs = nx; const stem = nxs * nxs / 0.02 + ny2st + nz2st;
        let base = skull <= 1.0 ? 0.35 : 0.0;
        if (vent  <= 1.0) base = 0.08;
        if (gml   <= 1.0 && base < 0.75) base = 0.75;
        if (gmr   <= 1.0 && base < 0.75) base = 0.75;
        if (stem  <= 1.0 && base < 0.55) base = 0.55;

        const slice = new Float32Array(T * E);
        let i = 0;
        for (let t = 0; t < T; t++) {
            const ts = 1.0 + 0.15 * t / Math.max(1, T - 1);
            for (let e = 0; e < E; e++) {
                const v = base * ts * (1.0 + 0.08 * e / Math.max(1, E - 1));
                slice[i++] = v > 1.0 ? 1.0 : v;
            }
        }
        return slice;
    }
}
