// Demand-driven plane cache for the orthoslicer.
//
// The viewer never needs a whole volume, only the three planes through the cursor plus whatever the user is about to
// scroll to. So the renderer tells the store which planes are *wanted* (with priorities) and the store
//   · keeps decoded planes in a byte-budgeted LRU (CPU side),
//   · requests only wanted planes it does not have, most urgent first, a few at a time,
//   · batches neighbouring low-priority planes into one slab request ([z0:z1] etc.),
//   · aborts requests that are no longer wanted (the cursor moved on) and drops them from the queue.
// Decoding happens in the HastyClient workers (client.enableWorkers()), so the main thread only splits slabs.

export const VIEWS = ['axial', 'sagittal', 'coronal'];
/** axial: fixed z, sagittal: fixed x, coronal: fixed y */
export const VIEW_AXIS = { axial: 'z', sagittal: 'x', coronal: 'y' };

export const planeKey = (view, t, e, i) => `${view[0]}${t},${e},${i}`;

class Dims {
    _setShape(shape) {
        this._ndim = shape.length;                // actual server-side ndim
        const s = shape.slice();
        while (s.length < 5) s.unshift(1);
        this.shape = s;
    }
    get T() { return this.shape[0]; }
    get E() { return this.shape[1]; }
    get Z() { return this.shape[2]; }
    get Y() { return this.shape[3]; }
    get X() { return this.shape[4]; }
    total(view) { return view === 'axial' ? this.Z : view === 'sagittal' ? this.X : this.Y; }
}

/**
 * Wanted planes around `pos`, each {view, t, e, i, pri} (smaller pri = more urgent).
 *   radius   : spatial neighbours along each view's axis (the direction of the last motion goes first)
 *   tRadius / eRadius : neighbouring time points / echoes (their three current planes), for stepping through t and e
 */
export function computeWanted(pos, dims, { radius = 3, tRadius = 0, eRadius = 0, motion = {} } = {}) {
    const out = [];
    const add = (view, t, e, i, pri) => { if (i >= 0 && i < dims.total(view)) out.push({ view, t, e, i, pri }); };
    for (const view of VIEWS) {
        const ax = VIEW_AXIS[view], c = pos[ax], dir = motion[ax] || 0;
        add(view, pos.t, pos.e, c, 0);
        for (let r = 1; r <= radius; r++) {
            add(view, pos.t, pos.e, c + r, r + (dir < 0 ? 0.5 : 0));
            add(view, pos.t, pos.e, c - r, r + (dir > 0 ? 0.5 : 0));
        }
    }
    const around = (dt, de, pri) => {
        const t = pos.t + dt, e = pos.e + de;
        if (t < 0 || t >= dims.T || e < 0 || e >= dims.E) return;
        for (const view of VIEWS) add(view, t, e, pos[VIEW_AXIS[view]], pri);
    };
    for (let r = 1; r <= tRadius; r++) { around(r, 0, 30 + r); around(-r, 0, 31 + r); }
    for (let r = 1; r <= eRadius; r++) { around(0, r, 40 + r); around(0, -r, 41 + r); }
    return out;
}

// ─── PlaneStore: planes of a uint16 comprep tensor on the hasty server ───────────────────────────────────────────────

export class PlaneStore extends Dims {
    /**
     * @param {import('../../common/hasty_client.js').HastyClient} client
     * @param {string} id       value id (hex) of the u16 display tensor
     * @param {number[]} shape  tensor shape (1–5 dims; missing leading dims are T, E)
     */
    constructor(client, id, shape, { dtype = 'i16', comprepConfig = null, budgetBytes = 512 << 20, concurrency = 6, maxBatch = 4 } = {}) {
        super();
        this._setShape(shape);
        this.client = client;
        this.id = id;
        this.dtype = dtype;
        this.comprepConfig = comprepConfig;
        this.budget = budgetBytes;
        this.concurrency = concurrency;
        this.maxBatch = maxBatch;
        /** @type {Map<string, Uint16Array>} insertion order = LRU order */
        this.cache = new Map();
        this.bytes = 0;
        this.wanted = new Map();
        this.queue = [];
        /** @type {Map<string, {ac: AbortController, keys: string[], pri: number}>} request id → in flight */
        this.inflight = new Map();
        this._failed = new Map();
        this._nextReq = 0;
        this.stats = { requests: 0, planes: 0, aborted: 0, bytes: 0 };
        /** called (view, t, e, i, data) when a plane arrived */
        this.onPlane = null;
        /** called when extra-view data arrived */
        this.onExtra = null;
        this._extra = new Map();
        this._extraTimer = 0;
        this._extraAc = null;
        this._extraWant = null;
    }

    get queued() { return this.queue.length; }
    get active() { return this.inflight.size; }

    peek(view, t, e, i) {
        const k = planeKey(view, t, e, i);
        const d = this.cache.get(k);
        if (d) { this.cache.delete(k); this.cache.set(k, d); }
        return d ?? null;
    }

    has(view, t, e, i) { return this.cache.has(planeKey(view, t, e, i)); }

    /** replace the wanted set; cancels requests nobody wants any more and re-prioritises the queue */
    setWanted(items) {
        this.wanted = new Map();
        for (const it of items) this.wanted.set(planeKey(it.view, it.t, it.e, it.i), it);
        for (const [rid, f] of this.inflight) {
            if (!f.keys.some((k) => this.wanted.has(k))) { f.ac.abort(); this.stats.aborted++; this.inflight.delete(rid); }
        }
        const busy = new Set();
        for (const f of this.inflight.values()) for (const k of f.keys) busy.add(k);
        const now = performance.now();
        const q = [];
        for (const [k, it] of this.wanted) {
            if (this.cache.has(k) || busy.has(k)) continue;
            if ((this._failed.get(k) ?? 0) > now) continue;
            q.push(it);
        }
        q.sort((a, b) => a.pri - b.pri);
        this.queue = q;
        this._pump();
    }

    _pump() {
        while (this.queue.length) {
            const head = this.queue[0];
            const limit = head.pri < 1 ? this.concurrency + 2 : this.concurrency;   // the planes on screen get a reserved lane
            if (this.inflight.size >= limit) return;
            this.queue.shift();
            let lo = head.i, hi = head.i;
            if (head.pri >= 1 && this.maxBatch > 1) {          // slab of adjacent low-priority planes in one request
                const idx = new Set();
                for (const it of this.queue) if (it.pri >= 1 && it.view === head.view && it.t === head.t && it.e === head.e) idx.add(it.i);
                while (hi - lo + 1 < this.maxBatch) {
                    if (idx.has(hi + 1)) hi++;
                    else if (idx.has(lo - 1)) lo--;
                    else break;
                }
                if (hi > lo) this.queue = this.queue.filter((it) => !(it.view === head.view && it.t === head.t && it.e === head.e && it.i >= lo && it.i <= hi));
            }
            this._fetch(head.view, head.t, head.e, lo, hi);
        }
    }

    _slice(parts5d) { return '[' + parts5d.slice(5 - this._ndim).join(',') + ']'; }

    _fetch(view, t, e, lo, hi) {
        const n = hi - lo + 1, r = `${lo}:${hi + 1}`;
        const slice = this._slice(view === 'axial' ? [t, e, r, ':', ':'] : view === 'sagittal' ? [t, e, ':', ':', r] : [t, e, ':', r, ':']);
        const keys = [];
        for (let i = lo; i <= hi; i++) keys.push(planeKey(view, t, e, i));
        const rid = ++this._nextReq;
        const ac = new AbortController();
        this.inflight.set(rid, { ac, keys });
        this.stats.requests++;
        this.client.getTensor(this.id, { slice, codec: 'lz4', delta: true, signal: ac.signal }).then((tensor) => {
            const d = tensor.data;
            if (tensor.shape.length !== 3 || tensor.shape[view === 'axial' ? 0 : view === 'sagittal' ? 2 : 1] !== n) throw new Error(`plane slab shape ${tensor.shape}`);
            this.stats.bytes += d.byteLength;
            const { Z, Y, X } = this;
            for (let j = 0; j < n; j++) {
                let plane;
                if (view === 'axial') plane = d.subarray(j * Y * X, (j + 1) * Y * X);
                else if (view === 'sagittal') {                              // [Z, Y, n] → Z×Y
                    if (n === 1) plane = d;
                    else { plane = new Uint16Array(Z * Y); for (let k = 0; k < Z * Y; k++) plane[k] = d[k * n + j]; }
                } else {                                                      // [Z, n, X] → Z×X
                    if (n === 1) plane = d;
                    else { plane = new Uint16Array(Z * X); for (let z = 0; z < Z; z++) plane.set(d.subarray((z * n + j) * X, (z * n + j + 1) * X), z * X); }
                }
                this._store(keys[j], plane);
                this.stats.planes++;
                this.onPlane?.(view, t, e, lo + j, plane);
            }
        }).catch((err) => {
            if (err?.name === 'AbortError' || ac.signal.aborted) return;
            const msg = err?.message ?? String(err);
            for (const k of keys) this._failed.set(k, performance.now() + 5000);
            if (/grpc 5/.test(msg)) { if (!this._missingLogged) { console.warn('PlaneStore: value not found — not retrying'); this._missingLogged = true; } }
            else console.error('PlaneStore fetch', slice, err);
        }).finally(() => { this.inflight.delete(rid); this._pump(); });
    }

    _store(k, plane) {
        const old = this.cache.get(k);
        if (old) { this.bytes -= old.byteLength; this.cache.delete(k); }
        this.cache.set(k, plane);
        this.bytes += plane.byteLength;
        if (this.bytes <= this.budget) return;
        for (const [key, d] of this.cache) {                                  // oldest first; never evict what is wanted
            if (this.bytes <= this.budget) break;
            if (this.wanted.has(key)) continue;
            this.cache.delete(key);
            this.bytes -= d.byteLength;
        }
    }

    // ── extra view: the (T×E) series at one voxel, debounced ────────────────────────────────────────────────────
    peekExtra(z, y, x) { return this._extra.get(`${z},${y},${x}`) ?? null; }

    requestExtra(z, y, x, delayMs = 80) {
        const key = `${z},${y},${x}`;
        this._extraWant = key;
        clearTimeout(this._extraTimer);
        if (this._extra.has(key)) return;
        this._extraTimer = setTimeout(() => {
            if (this._extraWant !== key) return;
            this._extraAc?.abort();
            const ac = this._extraAc = new AbortController();
            this.client.getTensor(this.id, { slice: this._slice([':', ':', z, y, x]), codec: 'lz4', delta: true, signal: ac.signal }).then((tensor) => {
                this._extra.set(key, tensor.data);
                if (this._extra.size > 64) this._extra.delete(this._extra.keys().next().value);
                this.onExtra?.();
            }).catch((err) => { if (err?.name !== 'AbortError' && !ac.signal.aborted) console.error('PlaneStore extra', err); });
        }, delayMs);
    }

    destroy() {
        for (const f of this.inflight.values()) f.ac.abort();
        this.inflight.clear();
        this._extraAc?.abort();
        clearTimeout(this._extraTimer);
    }
}

// ─── LazySource: the same interface over the synthetic LazyVolume (demo without ?id=) ────────────────────────────────
export class LazySource extends Dims {
    constructor(volume) {
        super();
        this._setShape(volume.shape);
        this.volume = volume;
        this.dtype = 'f32';
        this.comprepConfig = null;
        this.cache = new Map();
        this.onPlane = null;
        this.onExtra = null;
        this.stats = { requests: 0, planes: 0, aborted: 0, bytes: 0 };
        this.bytes = 0;
        this.queued = 0;
        this.active = 0;
    }
    peek(view, t, e, i) {
        const k = planeKey(view, t, e, i);
        let d = this.cache.get(k);
        if (d) { this.cache.delete(k); this.cache.set(k, d); return d; }
        d = view === 'axial' ? this.volume.getAxialSlice(t, e, i) : view === 'sagittal' ? this.volume.getSagittalSlice(t, e, i) : this.volume.getCoronalSlice(t, e, i);
        this.cache.set(k, d);
        if (this.cache.size > 256) this.cache.delete(this.cache.keys().next().value);
        return d;
    }
    setWanted() {}
    peekExtra(z, y, x) { return this.volume.getExtraSlice(z, y, x); }
    requestExtra() {}
    destroy() {}
}
