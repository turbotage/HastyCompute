// lbm_core.h — per-node physics of the cardiac lattice-Boltzmann solver (hasty-cfd), single source for both backends:
//   GPU: NVRTC compiles this header + lbm_cuda.cu (one __global__ wrapper per node function);
//   CPU: the C compiler builds lbm_cpu.c (loops over node ranges; the Rust side splits ranges over threads).
// No standard headers on the CUDA path. Lattice units throughout (dx = dt = 1, c_s² = 1/3); populations are stored
// shifted, g_i = f_i − w_i (better fp32 resolution around ρ = 1).
//
// Layout: the fluid band is covered by active 8³ tiles; node n = tile·512 + (lz·8 + ly)·8 + lx; per-population arrays
// are SoA, f[i·N + n]; tile_nbr[t·27 + (dz+1)·9 + (dy+1)·3 + (dx+1)] = neighbouring tile or −1.
//
// Physics:
// - D3Q19, regularized BGK (Latt & Chopard 2006: the non-equilibrium part projected on the second-order Hermite
//   moments) with the WALE eddy viscosity (Nicoud & Ducros 1999) from central differences of the previous step's
//   velocity;
// - moving walls: the reference signed distance φ0 of the blood domain pulled through the prescribed motion,
//   φ(x, t) = φ0(Φ_t⁻¹ x) (fixed-point inverse on the motion field), interpolated bounce-back (Bouzidi, Firdaouss &
//   Lallemand 2001) with the wall velocity term (Ladd 1994; Bouzidi's 1/q scaling for q ≥ ½), refill of nodes uncovered
//   by a receding wall at equilibrium (neighbours' density, wall velocity). Mass consistency of the moving walls (the
//   chamber flows *are* the wall volume changes) is a gate: the dilating closed tube's flux Q(x) = 2πRṘx is met to a
//   few % (tests.rs); exact stores of the wall mass were tried and rejected (stale stores on tangentially sliding walls
//   biased the bulk flow more than they fixed);
// - valves: an orifice plate with a time-varying elliptical hole, or kinematic leaflets rotating about their hinge
//   chords, with the exact solid velocity; the core (solid fraction ≥ 0.99) is solid (half-way bounce-back with the
//   leaflet velocity, moving like the walls), the edge cells are partially saturated (Noble & Torczynski 1998);
// - inlets / outlets: equilibrium nodes in a slab beyond moving cut planes (velocity inlet with the density, pressure
//   outlet with the velocity of the neighbour one step inward).

#ifdef __CUDACC__
#define HD __device__ __forceinline__
#define CARR __constant__ const
#define ATOMIC_ADD(p, v) atomicAdd((p), (v))
#define ATOMIC_ADD_SHARED(p, v) atomicAdd((p), (v))
#define ATOMIC_MAXF(p, v) atomicMax((int*)(p), __float_as_int(v))
#else
#include <math.h>
#define HD static inline
#define CARR static const
// thread-private targets on the CPU (per-tile accumulator cells, per-chunk diagnostics)
#define ATOMIC_ADD(p, v) (*(p) += (v))
#define ATOMIC_MAXF(p, v) (*(p) = (*(p) > (v) ? *(p) : (v)))
// targets shared between threads (the wall-mass store of solid nodes): compare-and-swap on the bits
static inline void atomic_add_shared(float* p, float v) {
    union { float f; unsigned int u; } o, n;
    unsigned int* ip = (unsigned int*)p;
    o.u = __atomic_load_n(ip, __ATOMIC_RELAXED);
    do {
        n.f = o.f + v;
    } while (!__atomic_compare_exchange_n(ip, &o.u, n.u, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}
#define ATOMIC_ADD_SHARED(p, v) atomic_add_shared((p), (v))
#endif

#define LBM_Q 19
#define TILE3 512
#define VALVE_STRIDE 320
#define CUT_STRIDE 32
#define MAX_REGIONS 32
#define ACC_STRIDE 30

// flags
#define F_FLUID 1u
#define F_WAS 2u
#define F_REFILL 4u
#define F_BC 8u
#define F_PSM 16u
#define F_WALL 32u

CARR int CX[19] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0};
CARR int CY[19] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1};
CARR int CZ[19] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1};
CARR float WQ[19] = {1.f / 3.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f, 1.f / 18.f,
                     1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f,
                     1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f};
CARR int OPP[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

// ── parameter blocks (mirrored by #[repr(C)] structs in hasty-cfd) ──────────────────────────────────────────────────

typedef struct {
    int M;          // motion grid voxels
    int idx[4];     // Catmull-Rom time samples
    float w[4];     // weights for U
    float d[4];     // weights for ∂U/∂t (1/s)
} MotionParams;

typedef struct {
    float o[3];     // world position of lattice node (0, 0, 0), m
    float dx;       // m
    float mo[3];    // motion grid origin, voxel centres at mo + (i + ½) mh, m
    float mh;
    int mn[3];      // motion grid dims (x, y, z)
    float so[3];    // reference SDF grid origin (same convention), m
    float sh;
    int sn[3];
    float vscale;   // m/s → lattice velocity (dt / dx)
    int iters;      // fixed-point iterations of the inverse motion
    int first;      // 1: cold start of the inverse (X = x)
    int nvalves;
    int ncuts;
    int ntiles;
    int regions;    // 1: a region map (u8, the sdf grid) is sampled at the reference position
} GeomParams;

typedef struct {
    int ntiles;
    float tau0;     // molecular relaxation time 3ν + ½
    float cwale;    // WALE constant C_w (0: no LES)
    float uclip;    // velocity magnitude clip (lattice), 0 = off
    int ncuts;
    int fct;        // output coarsening (1, 2, 4, 8): cells of fct³ nodes
    float vel_out;  // lattice → m/s for the accumulator
    float tau_b;    // bulk relaxation time of the trace of Π^neq (≤ ½: same as the shear τ)
    float stress_out;  // lattice stress → Pa for the accumulator (ρ0 (dx/dt)²)
    int pad;
} StepParams;

// spin history (relative longitudinal magnetization s = M_z / M0 of the blood, a passive scalar): saturation by the
// slab's RF, (1 − cos α(z)) per TR, and T1 recovery per step; α from a table along the lab z (az0 + i adz)
typedef struct {
    float dt_tr;    // dt / TR
    float e1_step;  // exp(−dt / T1)
    float o[3];     // lattice node 0 relative to the slab centre (lab, m)
    float nrm[3];   // slab normal (the profile table is over the signed distance along it)
    float dx;
    int ntab;
    float az0;
    float adz;
    int enabled;
} SpinParams;

// ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────────

HD float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

HD float h2f(unsigned short h) {
    unsigned int s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0u) v = ldexpf((float)m, -24);
    else if (e == 31u) v = 65504.f;
    else v = ldexpf((float)(m + 1024u), (int)e - 25);
    return s ? -v : v;
}

// neighbour of node n at offset (dx, dy, dz) ∈ {−1, 0, 1}³, −1 if its tile is not active
HD long long nbr_node(const int* tnbr, long long n, int dx, int dy, int dz) {
    long long t = n >> 9;
    int l = (int)(n & 511);
    int x = (l & 7) + dx, y = ((l >> 3) & 7) + dy, z = (l >> 6) + dz;
    int tx = x < 0 ? -1 : (x > 7 ? 1 : 0), ty = y < 0 ? -1 : (y > 7 ? 1 : 0), tz = z < 0 ? -1 : (z > 7 ? 1 : 0);
    long long tn = t;
    if (tx | ty | tz) {
        tn = tnbr[t * 27 + (tz + 1) * 9 + (ty + 1) * 3 + (tx + 1)];
        if (tn < 0) return -1;
    }
    return (tn << 9) | (long long)(((z & 7) << 6) | ((y & 7) << 3) | (x & 7));
}

// integer lattice coordinates of node n
HD void node_coords(const int* txyz, long long n, int* g) {
    long long t = n >> 9;
    int l = (int)(n & 511);
    g[0] = txyz[3 * t + 0] * 8 + (l & 7);
    g[1] = txyz[3 * t + 1] * 8 + ((l >> 3) & 7);
    g[2] = txyz[3 * t + 2] * 8 + (l >> 6);
}

// trilinear sample of component c of a SoA field F[c][nz][ny][nx] with voxel centres at o + (i + ½) h; zeros outside
// (per corner, as torch grid_sample with zeros padding)
HD float tri_zero(const float* F, int c, const int* nn, const float* o, float h, const float* X) {
    float gx = (X[0] - o[0]) / h - 0.5f, gy = (X[1] - o[1]) / h - 0.5f, gz = (X[2] - o[2]) / h - 0.5f;
    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;
    long long M = (long long)nn[0] * nn[1] * nn[2];
    float acc = 0.f;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                int x = ix + dx, y = iy + dy, z = iz + dz;
                if (x < 0 || y < 0 || z < 0 || x >= nn[0] || y >= nn[1] || z >= nn[2]) continue;
                float w = (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy) * (dz ? fz : 1.f - fz);
                acc += w * F[c * M + ((long long)z * nn[1] + y) * nn[0] + x];
            }
    return acc;
}

// trilinear sample (zeros outside) with its gradient d/dX (per metre)
HD float tri_zero_grad(const float* F, int c, const int* nn, const float* o, float h, const float* X, float* gr) {
    float gx = (X[0] - o[0]) / h - 0.5f, gy = (X[1] - o[1]) / h - 0.5f, gz = (X[2] - o[2]) / h - 0.5f;
    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;
    long long M = (long long)nn[0] * nn[1] * nn[2];
    float acc = 0.f;
    gr[0] = gr[1] = gr[2] = 0.f;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                int x = ix + dx, y = iy + dy, z = iz + dz;
                if (x < 0 || y < 0 || z < 0 || x >= nn[0] || y >= nn[1] || z >= nn[2]) continue;
                float wx = dx ? fx : 1.f - fx, wy = dy ? fy : 1.f - fy, wz = dz ? fz : 1.f - fz;
                float sx = dx ? 1.f : -1.f, sy = dy ? 1.f : -1.f, sz = dz ? 1.f : -1.f;
                float f = F[c * M + ((long long)z * nn[1] + y) * nn[0] + x];
                acc += wx * wy * wz * f;
                gr[0] += sx * wy * wz * f / h;
                gr[1] += wx * sy * wz * f / h;
                gr[2] += wx * wy * sz * f / h;
            }
    return acc;
}

// trilinear sample of a scalar grid, `outside` beyond half a voxel outside the grid (clamped inside)
HD float tri_clamp(const float* F, const int* nn, const float* o, float h, const float* X, float outside) {
    float gx = (X[0] - o[0]) / h - 0.5f, gy = (X[1] - o[1]) / h - 0.5f, gz = (X[2] - o[2]) / h - 0.5f;
    if (gx < -0.5f || gy < -0.5f || gz < -0.5f || gx > nn[0] - 0.5f || gy > nn[1] - 0.5f || gz > nn[2] - 0.5f) return outside;
    gx = gx < 0.f ? 0.f : (gx > nn[0] - 1 ? (float)(nn[0] - 1) : gx);
    gy = gy < 0.f ? 0.f : (gy > nn[1] - 1 ? (float)(nn[1] - 1) : gy);
    gz = gz < 0.f ? 0.f : (gz > nn[2] - 1 ? (float)(nn[2] - 1) : gz);
    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    if (ix > nn[0] - 2) ix = nn[0] - 2 < 0 ? 0 : nn[0] - 2;
    if (iy > nn[1] - 2) iy = nn[1] - 2 < 0 ? 0 : nn[1] - 2;
    if (iz > nn[2] - 2) iz = nn[2] - 2 < 0 ? 0 : nn[2] - 2;
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;
    float acc = 0.f;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) {
                int x = ix + dx < nn[0] ? ix + dx : nn[0] - 1, y = iy + dy < nn[1] ? iy + dy : nn[1] - 1, z = iz + dz < nn[2] ? iz + dz : nn[2] - 1;
                float w = (dx ? fx : 1.f - fx) * (dy ? fy : 1.f - fy) * (dz ? fz : 1.f - fz);
                acc += w * F[((long long)z * nn[1] + y) * nn[0] + x];
            }
    return acc;
}

// shifted equilibrium g_i^eq = f_i^eq − w_i
HD float geq(int i, float rho, float ux, float uy, float uz, float usq) {
    float cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
    return WQ[i] * ((rho - 1.f) + rho * (3.f * cu + 4.5f * cu * cu - 1.5f * usq));
}

// ── motion: the field at the current time from the Catmull-Rom samples (U in m, ∂U/∂t in m/s) ────────────────────────

HD void node_motion(const MotionParams mp, long long k, const unsigned short* Uall, float* Ut, float* Vt) {
    long long M = mp.M;
    for (int c = 0; c < 3; c++) {
        float u = 0.f, v = 0.f;
        for (int j = 0; j < 4; j++) {
            float s = h2f(Uall[((long long)mp.idx[j] * 3 + c) * M + k]);
            u += mp.w[j] * s;
            v += mp.d[j] * s;
        }
        Ut[c * M + k] = u * 1e-3f;
        Vt[c * M + k] = v * 1e-3f;
    }
}

// ── valves ──────────────────────────────────────────────────────────────────────────────────────────────────────────
// record (VALVE_STRIDE floats, positions and lengths in lattice units relative to node (0, 0, 0)):
//   0–2 centre c, 3–5 n (downstream), 6–8 e1, 9–11 e2, 12 R (hinge circle), 13 R_out (annulus ring), 14 thickness,
//   15 kind (0 orifice, 1 leaflets), 16 opening (0..1), 17 θ, 18 ω (rad per step), 19–21 annulus velocity,
//   22 number of leaflets (≤ 16: strips of the anatomical leaflets, see hasty-cfd case::presets), 23–24 orifice hole
//   semi-axes / R (at opening 1), 25 reach (culling radius), 26 frame (1: tissue — centre and axes in the reference
//   pose, evaluated at the node's reference position, the tissue velocity added to the solid velocity; 0: rigid);
//   27–29 annulus-plane centre (lab), 30–32 its normal, 33 its radius (the rigid closed disk of a tissue-frame valve);
//   leaflet k at 64 + 16k: 0–1 hinge point Pa (valve-local 2D), 2–3 hinge direction d (2D; d3 = m0 × n),
//   4–5 inward direction m0 (2D), 6 vertex count, 8–15 polygon (a, b) in leaflet coordinates (a along d, b along m0),
//   counter-clockwise, vertex 0 = Pa.

HD float seg_dist(float px, float py, float ax, float ay, float bx, float by) {
    float ex = bx - ax, ey = by - ay, wx = px - ax, wy = py - ay;
    float l2 = ex * ex + ey * ey;
    float s = l2 > 0.f ? (wx * ex + wy * ey) / l2 : 0.f;
    s = clamp01(s);
    float dx = wx - s * ex, dy = wy - s * ey;
    return sqrtf(dx * dx + dy * dy);
}

HD float poly_dist(const float* P, int nv, float a, float b) {
    int inside = 1;
    float dmin = 1e30f;
    for (int k = 0; k < nv; k++) {
        float ax = P[2 * k], ay = P[2 * k + 1];
        float bx = P[2 * ((k + 1) % nv)], by = P[2 * ((k + 1) % nv) + 1];
        if ((bx - ax) * (b - ay) - (by - ay) * (a - ax) < 0.f) inside = 0;
        float d = seg_dist(a, b, ax, ay, bx, by);
        dmin = d < dmin ? d : dmin;
    }
    return inside ? 0.f : dmin;
}

// solid fraction of the valve at lattice point g and the solid velocity there
// lab length of a reference-frame unit vector a under F = I + ∇U (row-major; F null: rigid frame, 1)
HD float stretch(const float* F, const float* a) {
    if (!F) return 1.f;
    float b0 = F[0] * a[0] + F[1] * a[1] + F[2] * a[2], b1 = F[3] * a[0] + F[4] * a[1] + F[5] * a[2], b2 = F[6] * a[0] + F[7] * a[1] + F[8] * a[2];
    return sqrtf(b0 * b0 + b1 * b1 + b2 * b2);
}

HD float valve_solid(const float* V, const float* g, const float* glab, float* us, const float* F) {
    float r[3] = {g[0] - V[0], g[1] - V[1], g[2] - V[2]};
    float rr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    if (rr > V[25] * V[25]) return 0.f;
    // distances across the valve sheet in lab lattice units: a tissue-frame valve sits in the blood, whose motion
    // field compresses it up to ~3× along the normal at end-systole; leaflets keep their thickness
    float sn = stretch(F, V + 3);
    float zl = (r[0] * V[3] + r[1] * V[4] + r[2] * V[5]) * sn;
    float xl = r[0] * V[6] + r[1] * V[7] + r[2] * V[8];
    float yl = r[0] * V[9] + r[1] * V[10] + r[2] * V[11];
    float rho = sqrtf(xl * xl + yl * yl);
    float R = V[12], Ro = V[13], th = V[14];
    us[0] = V[19];
    us[1] = V[20];
    us[2] = V[21];
    float bz = clamp01(0.5f + 0.5f * th - fabsf(zl));
    float B = 0.f;
    if (V[15] < 0.5f) {  // orifice plate with an elliptical hole
        if (bz > 0.f) {
            float br = clamp01(0.5f + Ro - rho);
            float bh = 1.f;
            float ax = V[16] * V[23] * R, ay = V[16] * V[24] * R;
            if (ax > 0.25f && ay > 0.25f) {
                float re = sqrtf((xl / ax) * (xl / ax) + (yl / ay) * (yl / ay));
                bh = clamp01(0.5f + (re - 1.f) * (ax < ay ? ax : ay));
            }
            B = bz * br * bh;
        }
        return B;
    }
    // leaflets: the fixed ring R ≤ ρ ≤ R_out and the circular segments beyond each hinge chord (in the annulus plane)
    int nl = (int)V[22];
    if (bz > 0.f) {
        B = bz * clamp01(0.5f + Ro - rho) * clamp01(0.5f + rho - R);
        for (int k = 0; k < nl; k++) {
            const float* L = V + 64 + 16 * k;
            float b0 = (xl - L[0]) * L[4] + (yl - L[1]) * L[5];
            // reaching ½ cell beyond the hinge circle: ring and segment would each give B = ½ at ρ = R, a porous seam
            // around the whole annulus (closed AV valves leaked through it in the moving heart)
            float bs = bz * clamp01(0.5f - b0) * clamp01(1.f + R - rho);
            B = bs > B ? bs : B;
        }
    }
    // closed (opening < 0.05): a full solid disk, at least 4 lattice units thick, fading out by opening 0.05. The closed
    // strip fans do not tile the disk exactly (commissures, the coaptation line): their partial cells bridged atrium
    // and ventricle in the moving heart, and a partially saturated sheet passes most of the flow
    if (V[16] < 0.05f) {
        float thc = th > 4.f ? th : 4.f;
        float fade = 1.f - V[16] / 0.05f;
        float bd = clamp01(0.5f + 0.5f * thc - fabsf(zl)) * clamp01(1.f + R - rho) * fade;
        if (bd > B) B = bd;
        // tissue frame: also a rigid disk on the plane fitted to the moved annulus, reaching 3 mm into the collar and
        // walls (the valve centre sits in the cavity's motion field, which squeezes and shears it at end-systole)
        if (F && V[33] > 0.f) {
            float q0 = glab[0] - V[27], q1 = glab[1] - V[28], q2 = glab[2] - V[29];
            float zr = q0 * V[30] + q1 * V[31] + q2 * V[32];
            float l0 = q0 - zr * V[30], l1 = q1 - zr * V[31], l2 = q2 - zr * V[32];
            float rr_ = sqrtf(l0 * l0 + l1 * l1 + l2 * l2);
            float br = clamp01(0.5f + 0.5f * thc - fabsf(zr)) * clamp01(0.5f + V[33] - rr_) * fade;
            if (br > B) B = br;
        }
    }
    float ct = cosf(V[17]), st = sinf(V[17]);
    for (int k = 0; k < nl; k++) {
        const float* L = V + 64 + 16 * k;
        float pa[3], d3[3], m3[3], mp[3], np[3];
        for (int a = 0; a < 3; a++) {
            pa[a] = V[a] + L[0] * V[6 + a] + L[1] * V[9 + a];
            d3[a] = L[2] * V[6 + a] + L[3] * V[9 + a];
            m3[a] = L[4] * V[6 + a] + L[5] * V[9 + a];
            mp[a] = ct * m3[a] + st * V[3 + a];
            np[a] = -st * m3[a] + ct * V[3 + a];
        }
        float v[3] = {g[0] - pa[0], g[1] - pa[1], g[2] - pa[2]};
        float a_ = v[0] * d3[0] + v[1] * d3[1] + v[2] * d3[2];
        float b_ = v[0] * mp[0] + v[1] * mp[1] + v[2] * mp[2];
        float h_ = (v[0] * np[0] + v[1] * np[1] + v[2] * np[2]) * stretch(F, np);
        float din = poly_dist(L + 8, (int)L[6], a_, b_);
        float D = sqrtf(din * din + h_ * h_);
        float bl = clamp01(0.5f + 0.5f * th - D);
        if (bl > B) {
            B = bl;
            float w = V[18];
            us[0] = V[19] + w * (d3[1] * v[2] - d3[2] * v[1]);
            us[1] = V[20] + w * (d3[2] * v[0] - d3[0] * v[2]);
            us[2] = V[21] + w * (d3[0] * v[1] - d3[1] * v[0]);
        }
    }
    return B;
}

// ── cuts (inlets / outlets) ─────────────────────────────────────────────────────────────────────────────────────────
// record (CUT_STRIDE floats, lattice units): 0–2 centre, 3–5 outward unit normal, 6 radius, 7 slab length beyond the
// plane, 8 kind (0 velocity inlet, 1 pressure outlet), 9 outlet density, 10–12 inlet velocity (full vector),
// 13 sponge length inside the domain before the plane, 14 sponge strength (added τ at the plane, quadratic ramp),
// 16–18 the cut's own velocity (fluxes are measured relative to it), 24–26 reference-pose centre, 27–29 reference
// normal, 30 reference radius: membership (slab beyond the plane within 1.25 R, flux layer −1 < s ≤ 0 and sponge
// within 1.1 R) is decided at the node's reference position, so a neighbouring chamber the motion brings near a cut
// never joins its slab or its flux layer

// position of a reference point relative to the cut's reference plane: s along the normal, l² lateral
HD void cut_ref(const float* C, const float* gX, float* s, float* l2) {
    float r[3] = {gX[0] - C[24], gX[1] - C[25], gX[2] - C[26]};
    float s_ = r[0] * C[27] + r[1] * C[28] + r[2] * C[29];
    float a = r[0] - s_ * C[27], b = r[1] - s_ * C[28], c = r[2] - s_ * C[29];
    *s = s_;
    *l2 = a * a + b * b + c * c;
}

// ── geometry pass 1: inverse motion, distance, material velocity, valves, cuts ─────────────────────────────────────

HD void node_geom1(const GeomParams p, long long n, long long N, const int* txyz, const float* Ut, const float* Vt,
                   const float* sdf, const float* valves, const float* cuts, float* xref, float* phi, float* um,
                   float* bsol, unsigned char* bcid, unsigned char* flags, float* spg, unsigned char* pid,
                   const unsigned char* rmap, unsigned char* rid) {
    int gi[3];
    node_coords(txyz, n, gi);
    float g[3] = {(float)gi[0], (float)gi[1], (float)gi[2]};
    float x[3] = {p.o[0] + g[0] * p.dx, p.o[1] + g[1] * p.dx, p.o[2] + g[2] * p.dx};
    float X[3];
    if (p.first) {
        X[0] = x[0];
        X[1] = x[1];
        X[2] = x[2];
    } else {
        X[0] = xref[n];
        X[1] = xref[N + n];
        X[2] = xref[2 * N + n];
    }
    // Newton on X + U(X) = x (the fixed point X = x − U(X) diverges where ‖∇U‖ > 1: the cavities at end-systole reach
    // 1.35 in the phantom's motion); steps capped at one motion voxel, a fixed-point step where I + ∇U is near-singular
    float Fm[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
    for (int it = 0; it < p.iters; it++) {
        float u[3], J[3][3];
        for (int c = 0; c < 3; c++) u[c] = tri_zero_grad(Ut, c, p.mn, p.mo, p.mh, X, J[c]);
        float r0 = X[0] + u[0] - x[0], r1 = X[1] + u[1] - x[1], r2 = X[2] + u[2] - x[2];
        float a00 = 1.f + J[0][0], a01 = J[0][1], a02 = J[0][2];
        float a10 = J[1][0], a11 = 1.f + J[1][1], a12 = J[1][2];
        float a20 = J[2][0], a21 = J[2][1], a22 = 1.f + J[2][2];
        float c00 = a11 * a22 - a12 * a21, c01 = a02 * a21 - a01 * a22, c02 = a01 * a12 - a02 * a11;
        float c10 = a12 * a20 - a10 * a22, c11 = a00 * a22 - a02 * a20, c12 = a02 * a10 - a00 * a12;
        float c20 = a10 * a21 - a11 * a20, c21 = a01 * a20 - a00 * a21, c22 = a00 * a11 - a01 * a10;
        float det = a00 * c00 + a01 * c10 + a02 * c20;
        Fm[0] = a00; Fm[1] = a01; Fm[2] = a02;
        Fm[3] = a10; Fm[4] = a11; Fm[5] = a12;
        Fm[6] = a20; Fm[7] = a21; Fm[8] = a22;
        float d0, d1, d2;
        if (det > 0.02f) {
            float id = 1.f / det;
            d0 = (c00 * r0 + c01 * r1 + c02 * r2) * id;
            d1 = (c10 * r0 + c11 * r1 + c12 * r2) * id;
            d2 = (c20 * r0 + c21 * r1 + c22 * r2) * id;
        } else {
            d0 = r0;
            d1 = r1;
            d2 = r2;
        }
        float dn = sqrtf(d0 * d0 + d1 * d1 + d2 * d2);
        if (dn > p.mh) {
            float s_ = p.mh / dn;
            d0 *= s_;
            d1 *= s_;
            d2 *= s_;
        }
        X[0] -= d0;
        X[1] -= d1;
        X[2] -= d2;
    }
    xref[n] = X[0];
    xref[N + n] = X[1];
    xref[2 * N + n] = X[2];
    float ph = tri_clamp(sdf, p.sn, p.so, p.sh, X, -1.f) / p.dx;
    float v[3] = {tri_zero(Vt, 0, p.mn, p.mo, p.mh, X) * p.vscale, tri_zero(Vt, 1, p.mn, p.mo, p.mh, X) * p.vscale, tri_zero(Vt, 2, p.mn, p.mo, p.mh, X) * p.vscale};
    float B = 0.f;
    // reference position in lattice units (tissue-frame valves deform with the walls they seal against)
    float gX[3] = {(X[0] - p.o[0]) / p.dx, (X[1] - p.o[1]) / p.dx, (X[2] - p.o[2]) / p.dx};
    float vt[3] = {v[0], v[1], v[2]};
    int probe = 0;
    for (int k = 0; k < p.nvalves; k++) {
        float us[3];
        const float* VR = valves + k * VALVE_STRIDE;
        int tissue = VR[26] > 0.5f;
        // pressure probes of the valve: balls of radius 0.4 R centred 0.5 R + 4 cells up- and downstream of the disk
        // (reference frame for tissue valves), for the pressure-triggered opening
        if (!probe && ph > 0.f) {
            const float* q = tissue ? gX : g;
            float r0 = q[0] - VR[0], r1 = q[1] - VR[1], r2 = q[2] - VR[2];
            float z = r0 * VR[3] + r1 * VR[4] + r2 * VR[5];
            float l2 = r0 * r0 + r1 * r1 + r2 * r2 - z * z;
            float d = 0.5f * VR[12] + 4.f, rb = 0.4f * VR[12];
            if ((z + d) * (z + d) + l2 <= rb * rb) probe = 1 + 2 * k;
            else if ((z - d) * (z - d) + l2 <= rb * rb) probe = 2 + 2 * k;
        }
        float b = valve_solid(VR, tissue ? gX : g, g, us, tissue ? Fm : (const float*)0);
        if (tissue) {
            us[0] += vt[0];
            us[1] += vt[1];
            us[2] += vt[2];
        }
        if (b > B) {
            B = b;
            v[0] = us[0];
            v[1] = us[1];
            v[2] = us[2];
        }
    }
    int bc = 0, layer = 0;
    float sp = 0.f;
    if (ph > 0.f)
        for (int k = 0; k < p.ncuts; k++) {
            const float* C = cuts + k * CUT_STRIDE;
            float s_, l2;
            cut_ref(C, gX, &s_, &l2);
            float R2 = C[30] * C[30];
            if (!bc && s_ > 0.f && s_ <= C[7] + 1.f && l2 <= 1.5625f * R2) bc = k + 1;
            // flux layer one reference unit thick (the host divides by its lab thickness 1 / |F⁻ᵀ n|)
            if (!layer && s_ > -1.f && s_ <= 0.f && l2 <= 1.21f * R2) layer = k + 1;
            // sponge: extra relaxation time over the last C[13] lattice units before an outlet plane (damps jets and
            // vortices before the equilibrium outlet nodes)
            if (C[13] > 0.f && s_ > -C[13] && l2 <= 1.21f * R2) {
                float w = s_ >= 0.f ? 1.f : 1.f + s_ / C[13];
                float a = C[14] * w * w;
                sp = a > sp ? a : sp;
            }
        }
    spg[n] = sp;
    // the valve core (B ≥ 0.99) is solid (bounce-back with the leaflet velocity: a sheet of partially saturated cells
    // stays porous, its rest equilibrium leaks through), the partial edge cells are partially saturated fluid
    int fluid = ph > 0.f && B < 0.99f;
    phi[n] = ph;
    um[n] = v[0];
    um[N + n] = v[1];
    um[2 * N + n] = v[2];
    bsol[n] = B;
    // slab id (with F_BC), else 128 + flux-layer id (0: none)
    bcid[n] = (unsigned char)(bc ? bc : (layer ? 127 + layer : 0));
    pid[n] = (unsigned char)(fluid ? probe : 0);
    // region of the reference position (nearest voxel of the builder's map; 0 outside / none)
    unsigned char rg = 0;
    if (p.regions && fluid) {
        int ix = (int)floorf((X[0] - p.so[0]) / p.sh), iy = (int)floorf((X[1] - p.so[1]) / p.sh), iz = (int)floorf((X[2] - p.so[2]) / p.sh);
        if (ix >= 0 && iy >= 0 && iz >= 0 && ix < p.sn[0] && iy < p.sn[1] && iz < p.sn[2]) rg = rmap[((long long)iz * p.sn[1] + iy) * p.sn[0] + ix];
    }
    rid[n] = rg;
    unsigned char f = flags[n] & F_WAS;
    if (fluid) f |= F_FLUID;
    if (bc && fluid) f |= F_BC;
    if (B > 0.f && fluid) f |= F_PSM;
    flags[n] = f;
}

// ── geometry pass 2: wall links (q fractions), refill marks ────────────────────────────────────────────────────────

HD void node_geom2(long long n, long long N, const int* tnbr, const float* phi, unsigned int* links, unsigned char* qlink, unsigned char* flags) {
    unsigned char f = flags[n];
    if (!(f & F_FLUID)) {
        links[n] = 0u;
        flags[n] = f & (unsigned char)~(F_WALL | F_REFILL);
        return;
    }
    unsigned int L = 0u;
    float p0 = phi[n];
    for (int i = 1; i < LBM_Q; i++) {
        long long s = nbr_node(tnbr, n, -CX[i], -CY[i], -CZ[i]);
        float q;
        if (s < 0) q = 0.5f;
        else if (!(flags[s] & F_FLUID)) {
            float ps = phi[s];
            if (ps > 0.f) q = 0.5f;  // a valve-core node inside the blood: half-way
            else {
                q = p0 / (p0 - ps);
                q = q < 0.02f ? 0.02f : (q > 1.f ? 1.f : q);
            }
        } else continue;
        L |= 1u << i;
        qlink[(long long)(i - 1) * N + n] = (unsigned char)(q * 255.f + 0.5f);
    }
    links[n] = L;
    f &= (unsigned char)~(F_WALL | F_REFILL);
    if (L) f |= F_WALL;
    if (!(f & F_WAS)) f |= F_REFILL;
    flags[n] = f;
}

// ── refill (nodes uncovered by a receding wall): equilibrium at the neighbours' mean density and the wall velocity ─

HD void node_refill(long long n, long long N, const int* tnbr, const unsigned char* flags, const float* um, float* f, float* ru) {
    unsigned char fl = flags[n];
    if (!((fl & F_REFILL) && (fl & F_FLUID))) return;
    float rs = 0.f;
    int cnt = 0;
    for (int i = 1; i < LBM_Q; i++) {
        long long s = nbr_node(tnbr, n, CX[i], CY[i], CZ[i]);
        if (s >= 0 && (flags[s] & F_FLUID) && (flags[s] & F_WAS)) {
            rs += ru[s];
            cnt++;
        }
    }
    float rho = cnt ? rs / cnt : 1.f;
    float ux = um[n], uy = um[N + n], uz = um[2 * N + n];
    float usq = ux * ux + uy * uy + uz * uz;
    for (int i = 0; i < LBM_Q; i++) f[(long long)i * N + n] = geq(i, rho, ux, uy, uz, usq);
    ru[n] = rho;
    ru[N + n] = ux;
    ru[2 * N + n] = uy;
    ru[3 * N + n] = uz;
}

// a node the moving wall uncovered: s from its old fluid neighbours (as the density in node_refill)
HD void node_srefill(long long n, const int* tnbr, const unsigned char* flags, float* s) {
    unsigned char fl = flags[n];
    if (!((fl & F_REFILL) && (fl & F_FLUID))) return;
    float acc = 0.f;
    int cnt = 0;
    for (int i = 1; i < LBM_Q; i++) {
        long long m = nbr_node(tnbr, n, CX[i], CY[i], CZ[i]);
        if (m >= 0 && (flags[m] & F_FLUID) && (flags[m] & F_WAS)) {
            acc += s[m];
            cnt++;
        }
    }
    s[n] = cnt ? acc / cnt : 1.f;
}

// semi-Lagrangian sample of field F at node n displaced by d (lattice units, |d| < 1): trilinear over the 8 surrounding
// nodes, a non-fluid corner taking the node's own value; the corners' min / max for the limiter
HD float sl_sample(long long n, const int* tnbr, const unsigned char* flags, const float* F, float dx_, float dy_, float dz_, float* lo, float* hi) {
    float b[3] = {dx_, dy_, dz_};
    int i0[3];
    float fr[3];
    for (int a = 0; a < 3; a++) {
        float fb = floorf(b[a]);
        i0[a] = (int)fb;
        fr[a] = b[a] - fb;
    }
    float self = F[n], v = 0.f, mn = 1e30f, mx = -1e30f;
    for (int dz = 0; dz < 2; dz++)
        for (int dy = 0; dy < 2; dy++)
            for (int dxx = 0; dxx < 2; dxx++) {
                int ox = i0[0] + dxx, oy = i0[1] + dy, oz = i0[2] + dz;
                float w = (dxx ? fr[0] : 1.f - fr[0]) * (dy ? fr[1] : 1.f - fr[1]) * (dz ? fr[2] : 1.f - fr[2]);
                float val = self;
                if (ox != 0 || oy != 0 || oz != 0) {
                    long long m = nbr_node(tnbr, n, ox, oy, oz);
                    if (m >= 0 && (flags[m] & F_FLUID)) val = F[m];
                }
                v += w * val;
                mn = fminf(mn, val);
                mx = fmaxf(mx, val);
            }
    if (lo) *lo = mn;
    if (hi) *hi = mx;
    return v;
}

// one step of the spin scalar, BFECC (back-and-forth error compensation) on semi-Lagrangian traces with this step's
// velocity: pass 0  ŝ = SL(s, −u); pass 1  s_c = s + ½ (s − SL(ŝ, +u)); pass 2  s' = SL(s_c, −u) clamped to the corner
// range of SL(s, −u), then the RF saturation of the slab and the T1 recovery; inlet slab nodes hold the cut's entering s
// (cut slot 19). A single first-order trace diffuses like upwind (D ≈ u Δx / 2: centimetres over a chamber residence)
HD void node_scalar(const SpinParams sp, int pass, long long n, long long N, const int* txyz, const int* tnbr, const unsigned char* flags, const unsigned char* bcid,
                    const float* cuts, const float* ru, const float* atab, const float* s_old, const float* aux, float* s_new) {
    unsigned char fl = flags[n];
    if (!(fl & F_FLUID)) {
        s_new[n] = s_old[n];
        return;
    }
    float ux = ru[N + n], uy = ru[2 * N + n], uz = ru[3 * N + n];
    if (fl & F_BC) {
        // velocity cuts hold the entering s while the flow enters (a cut with a prescribed outflow advects)
        const float* C = cuts + (bcid[n] - 1) * CUT_STRIDE;
        if (C[8] < 0.5f && (ux - C[16]) * C[3] + (uy - C[17]) * C[4] + (uz - C[18]) * C[5] <= 0.f) {
            s_new[n] = C[19];
            return;
        }
    }
    if (pass == 0) {
        s_new[n] = sl_sample(n, tnbr, flags, s_old, -ux, -uy, -uz, 0, 0);
        return;
    }
    if (pass == 1) {
        float back = sl_sample(n, tnbr, flags, aux, ux, uy, uz, 0, 0);
        s_new[n] = s_old[n] + 0.5f * (s_old[n] - back);
        return;
    }
    float lo, hi;
    sl_sample(n, tnbr, flags, s_old, -ux, -uy, -uz, &lo, &hi);
    float v = sl_sample(n, tnbr, flags, aux, -ux, -uy, -uz, 0, 0);
    v = fminf(fmaxf(v, lo), hi);
    int gi[3];
    node_coords(txyz, n, gi);
    float z = sp.nrm[0] * (sp.o[0] + gi[0] * sp.dx) + sp.nrm[1] * (sp.o[1] + gi[1] * sp.dx) + sp.nrm[2] * (sp.o[2] + gi[2] * sp.dx);
    float fa = (z - sp.az0) / sp.adz, al = 0.f;
    if (fa >= 0.f && fa <= sp.ntab - 1.001f) {
        int ia = (int)fa;
        float aa = fa - ia;
        al = atab[ia] * (1.f - aa) + atab[ia + 1] * aa;
    }
    v *= 1.f - (1.f - cosf(al)) * sp.dt_tr;
    v = 1.f - (1.f - v) * sp.e1_step;
    s_new[n] = fminf(fmaxf(v, 0.f), 1.f);
}

HD void node_commit(long long n, unsigned char* flags) {
    unsigned char f = flags[n];
    f &= (unsigned char)~(F_REFILL | F_WAS);
    if (f & F_FLUID) f |= F_WAS;
    flags[n] = f;
}

// ── WALE eddy viscosity from the previous step's velocity (lattice units, Δ = 1) ─────────────────────────────────────

HD float wale_nut(long long n, long long N, const int* tnbr, const unsigned char* flags, const float* ru, const float* um, float cw) {
    float gr[3][3];  // gr[a][b] = ∂u_a / ∂x_b
    for (int b = 0; b < 3; b++) {
        float up[3], dn[3];
        for (int sg = 0; sg < 2; sg++) {
            int o[3] = {0, 0, 0};
            o[b] = sg ? 1 : -1;
            long long s = nbr_node(tnbr, n, o[0], o[1], o[2]);
            float* u = sg ? up : dn;
            if (s >= 0 && (flags[s] & F_FLUID)) {
                u[0] = ru[N + s];
                u[1] = ru[2 * N + s];
                u[2] = ru[3 * N + s];
            } else if (s >= 0) {
                u[0] = um[s];
                u[1] = um[N + s];
                u[2] = um[2 * N + s];
            } else {
                u[0] = ru[N + n];
                u[1] = ru[2 * N + n];
                u[2] = ru[3 * N + n];
            }
        }
        for (int a = 0; a < 3; a++) gr[a][b] = 0.5f * (up[a] - dn[a]);
    }
    float g2[3][3];
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) g2[a][b] = gr[a][0] * gr[0][b] + gr[a][1] * gr[1][b] + gr[a][2] * gr[2][b];
    float tr = (g2[0][0] + g2[1][1] + g2[2][2]) / 3.f;
    float ss = 0.f, sd = 0.f;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
            float s = 0.5f * (gr[a][b] + gr[b][a]);
            float d = 0.5f * (g2[a][b] + g2[b][a]) - (a == b ? tr : 0.f);
            ss += s * s;
            sd += d * d;
        }
    float num = sd * sqrtf(sd);
    float den = ss * ss * sqrtf(ss) + sqrtf(sqrtf(sd * sd * sd * sd * sd));
    return den > 1e-30f ? cw * cw * num / den : 0.f;
}

// ── stream (pull) + collide ─────────────────────────────────────────────────────────────────────────────────────────

HD void node_step(const StepParams p, long long n, long long N, const int* tnbr, const unsigned char* flags, const unsigned int* links,
                  const unsigned char* qlink, const float* um, const float* bsol, const unsigned char* bcid, const float* cuts,
                  const float* src, float* dst, const float* ru_prev, float* ru_cur, const float* spg) {
    unsigned char fl = flags[n];
    if (!(fl & F_FLUID)) return;
    float g[LBM_Q];
    unsigned int L = links[n];
    g[0] = src[n];
    for (int i = 1; i < LBM_Q; i++) {
        if ((L >> i) & 1u) continue;
        long long s = nbr_node(tnbr, n, -CX[i], -CY[i], -CZ[i]);
        g[i] = src[(long long)i * N + s];
    }
    if (L) {
        float rw = ru_prev[n];
        rw = rw < 0.5f ? 0.5f : (rw > 1.5f ? 1.5f : rw);
        for (int i = 1; i < LBM_Q; i++) {
            if (!((L >> i) & 1u)) continue;
            int ib = OPP[i];
            float q = qlink[(long long)(i - 1) * N + n] * (1.f / 255.f);
            long long s = nbr_node(tnbr, n, -CX[i], -CY[i], -CZ[i]);
            float uw0 = um[n], uw1 = um[N + n], uw2 = um[2 * N + n];
            if (s >= 0) {
                uw0 = (1.f - q) * uw0 + q * um[s];
                uw1 = (1.f - q) * uw1 + q * um[N + s];
                uw2 = (1.f - q) * uw2 + q * um[2 * N + s];
            }
            float cu = CX[i] * uw0 + CY[i] * uw1 + CZ[i] * uw2;
            float fb = src[(long long)ib * N + n];
            float val;
            if (q < 0.5f) {
                long long x2 = nbr_node(tnbr, n, CX[i], CY[i], CZ[i]);
                if (x2 >= 0 && (flags[x2] & F_FLUID)) val = 2.f * q * fb + (1.f - 2.f * q) * src[(long long)ib * N + x2] + 6.f * WQ[i] * rw * cu;
                else val = fb + 6.f * WQ[i] * rw * cu;
            } else {
                float a = 0.5f / q;
                val = a * fb + (1.f - a) * src[(long long)i * N + n] + 3.f / q * WQ[i] * rw * cu;
            }
            g[i] = val;
        }
    }
    float rho = 1.f, jx = 0.f, jy = 0.f, jz = 0.f;
    for (int i = 0; i < LBM_Q; i++) {
        rho += g[i];
        jx += CX[i] * g[i];
        jy += CY[i] * g[i];
        jz += CZ[i] * g[i];
    }
    float ux = jx / rho, uy = jy / rho, uz = jz / rho;
    if (fl & F_BC) {
        // equilibrium node; the free quantity comes from the neighbour one step inward (−n), as in the
        // extrapolation boundary conditions (Guo et al. 2002): inlets take ρ from the interior, outlets take u
        const float* C = cuts + (bcid[n] - 1) * CUT_STRIDE;
        int ib = 1;
        float best = -2.f;
        for (int i = 1; i < LBM_Q; i++) {
            float cl = sqrtf((float)(CX[i] * CX[i] + CY[i] * CY[i] + CZ[i] * CZ[i]));
            float d = -(CX[i] * C[3] + CY[i] * C[4] + CZ[i] * C[5]) / cl;
            if (d > best) {
                best = d;
                ib = i;
            }
        }
        long long si = nbr_node(tnbr, n, CX[ib], CY[ib], CZ[ib]);
        float ri = rho, ui0 = ux, ui1 = uy, ui2 = uz;
        if (si >= 0 && (flags[si] & F_FLUID)) {
            ri = ru_prev[si];
            ui0 = ru_prev[N + si];
            ui1 = ru_prev[2 * N + si];
            ui2 = ru_prev[3 * N + si];
        }
        if (C[8] < 0.5f) {
            rho = ri;
            ux = C[10];
            uy = C[11];
            uz = C[12];
        } else {
            rho = C[9];
            ux = ui0;
            uy = ui1;
            uz = ui2;
        }
        float usq = ux * ux + uy * uy + uz * uz;
        for (int i = 0; i < LBM_Q; i++) dst[(long long)i * N + n] = geq(i, rho, ux, uy, uz, usq);
        ru_cur[n] = rho;
        ru_cur[N + n] = ux;
        ru_cur[2 * N + n] = uy;
        ru_cur[3 * N + n] = uz;
        return;
    }
    if (p.uclip > 0.f) {
        float um2 = ux * ux + uy * uy + uz * uz;
        if (um2 > p.uclip * p.uclip) {
            float s = p.uclip / sqrtf(um2);
            ux *= s;
            uy *= s;
            uz *= s;
        }
    }
    float usq = ux * ux + uy * uy + uz * uz;
    float ge[LBM_Q];
    float pxx = 0.f, pyy = 0.f, pzz = 0.f, pxy = 0.f, pxz = 0.f, pyz = 0.f;
    for (int i = 0; i < LBM_Q; i++) {
        ge[i] = geq(i, rho, ux, uy, uz, usq);
        float d = g[i] - ge[i];
        pxx += CX[i] * CX[i] * d;
        pyy += CY[i] * CY[i] * d;
        pzz += CZ[i] * CZ[i] * d;
        pxy += CX[i] * CY[i] * d;
        pxz += CX[i] * CZ[i] * d;
        pyz += CY[i] * CZ[i] * d;
    }
    float tau = p.tau0 + spg[n];
    if (p.cwale > 0.f) tau += 3.f * wale_nut(n, N, tnbr, flags, ru_prev, um, p.cwale);
    float keep = 1.f - 1.f / tau;
    // the trace of Π^neq (compression: sound) relaxes with its own τ_b — a bulk viscosity ν_b = (2/9)(τ_b − ½) that
    // damps the acoustic standing waves of the chambers (at τ0 ≈ 0.5008 they ring for whole cycles, ±40 mmHg), the
    // deviatoric part (shear) keeps τ
    float keep_b = p.tau_b > 0.5f ? 1.f - 1.f / p.tau_b : keep;
    float trp = (pxx + pyy + pzz) / 3.f;
    float gp[LBM_Q];
    for (int i = 0; i < LBM_Q; i++) {
        float c2 = (float)(CX[i] * CX[i] + CY[i] * CY[i] + CZ[i] * CZ[i]);
        float qd = CX[i] * CX[i] * pxx + CY[i] * CY[i] * pyy + CZ[i] * CZ[i] * pzz - trp * c2
                 + 2.f * (CX[i] * CY[i] * pxy + CX[i] * CZ[i] * pxz + CY[i] * CZ[i] * pyz);
        float qb = trp * (c2 - 1.f);
        gp[i] = ge[i] + 4.5f * WQ[i] * (keep * qd + keep_b * qb);
    }
    if (fl & F_PSM) {
        float B = bsol[n];
        float t5 = tau - 0.5f;
        float Bn = B * t5 / ((1.f - B) + t5);
        float sx = um[n], sy = um[N + n], sz = um[2 * N + n];
        float ssq = sx * sx + sy * sy + sz * sz;
        for (int i = 0; i < LBM_Q; i++) {
            int ib = OPP[i];
            float oms = (g[ib] - ge[ib]) - (g[i] - geq(i, rho, sx, sy, sz, ssq));
            gp[i] = g[i] + (1.f - Bn) * (gp[i] - g[i]) + Bn * oms;
        }
    }
    for (int i = 0; i < LBM_Q; i++) dst[(long long)i * N + n] = gp[i];
    ru_cur[n] = rho;
    ru_cur[N + n] = ux;
    ru_cur[2 * N + n] = uy;
    ru_cur[3 * N + n] = uz;
}

// ── output accumulation: Σw, Σw u, Σw u uᵀ per output cell (fct³ nodes) of the current phase bin, in m/s ──────────────

// … and at wall nodes (fluid next to a wall or a leaflet, not partially saturated) the wall shear stress, Pa:
// σ = −(τ0 − ½)/τ Π^neq_dev of the pre-collision populations (molecular viscosity; recovered from the post-collision ones: the regularized
// collision scales the deviatoric part by 1 − 1/τ; τ = τ0 + sponge, the WALE share neglected — it vanishes at walls),
// the wall normal n = ∇φ/|∇φ| (into the fluid), τ_w = σn − (n·σn) n; slots 10 Σw_wall, 11–13 Σ τ_w, 14 Σ |τ_w|
HD void node_accum(const StepParams p, long long n, long long N, const unsigned char* flags, const float* bsol, const float* ru, float* acc,
                   const int* tnbr, const float* phi, const float* spg, const float* pop, const float* sc) {
    unsigned char fl = flags[n];
    if (!(fl & F_FLUID) || (fl & F_BC)) return;
    float w = (fl & F_PSM) ? 1.f - bsol[n] : 1.f;
    if (w <= 0.f) return;
    long long t = n >> 9;
    int l = (int)(n & 511);
    int f = p.fct, c1 = 8 / p.fct;
    int cell = (((l >> 6) / f) * c1 + ((l >> 3) & 7) / f) * c1 + (l & 7) / f;
    float* a = acc + (t * (c1 * c1 * c1) + cell) * ACC_STRIDE;
    float ux = ru[N + n] * p.vel_out, uy = ru[2 * N + n] * p.vel_out, uz = ru[3 * N + n] * p.vel_out;
    ATOMIC_ADD(a + 0, w);
    ATOMIC_ADD(a + 1, w * ux);
    ATOMIC_ADD(a + 2, w * uy);
    ATOMIC_ADD(a + 3, w * uz);
    ATOMIC_ADD(a + 4, w * ux * ux);
    ATOMIC_ADD(a + 5, w * uy * uy);
    ATOMIC_ADD(a + 6, w * uz * uz);
    ATOMIC_ADD(a + 7, w * ux * uy);
    ATOMIC_ADD(a + 8, w * ux * uz);
    ATOMIC_ADD(a + 9, w * uy * uz);
    // third raw moments (slots 15–24: xxx yyy zzz xxy xxz xyy yyz xzz yzz xyz): the skewness of each cell's velocity
    // distribution, for the per-beat library (a phase-contrast phase is biased by asymmetric intravoxel distributions)
    ATOMIC_ADD(a + 15, w * ux * ux * ux);
    ATOMIC_ADD(a + 16, w * uy * uy * uy);
    ATOMIC_ADD(a + 17, w * uz * uz * uz);
    ATOMIC_ADD(a + 18, w * ux * ux * uy);
    ATOMIC_ADD(a + 19, w * ux * ux * uz);
    ATOMIC_ADD(a + 20, w * ux * uy * uy);
    ATOMIC_ADD(a + 21, w * uy * uy * uz);
    ATOMIC_ADD(a + 22, w * ux * uz * uz);
    ATOMIC_ADD(a + 23, w * uy * uz * uz);
    ATOMIC_ADD(a + 24, w * ux * uy * uz);
    // spin scalar (slots 25, 26: Σ w s, Σ w s²; 27–29: Σ w s u, the signal's correlation with the velocity inside
    // the cell; s = 1 when the spin history is off)
    float sv = sc ? sc[n] : 1.f;
    ATOMIC_ADD(a + 25, w * sv);
    ATOMIC_ADD(a + 26, w * sv * sv);
    ATOMIC_ADD(a + 27, w * sv * ux);
    ATOMIC_ADD(a + 28, w * sv * uy);
    ATOMIC_ADD(a + 29, w * sv * uz);
    if (!(fl & F_WALL) || (fl & F_PSM)) return;
    // wall normal from the distance field (central differences where both neighbours exist)
    float gr[3];
    for (int d = 0; d < 3; d++) {
        int ex = d == 0, ey = d == 1, ez = d == 2;
        long long sp = nbr_node(tnbr, n, ex, ey, ez), sm = nbr_node(tnbr, n, -ex, -ey, -ez);
        float pp = sp >= 0 ? phi[sp] : phi[n], pm = sm >= 0 ? phi[sm] : phi[n];
        float h2 = (sp >= 0 ? 1.f : 0.f) + (sm >= 0 ? 1.f : 0.f);
        gr[d] = h2 > 0.f ? (pp - pm) / h2 : 0.f;
    }
    float gl = sqrtf(gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]);
    if (gl < 1e-6f) return;
    float nx = gr[0] / gl, ny = gr[1] / gl, nz = gr[2] / gl;
    float rho = ru[n], lx = ru[N + n], ly = ru[2 * N + n], lz = ru[3 * N + n];
    float usq = lx * lx + ly * ly + lz * lz;
    float pxx = 0.f, pyy = 0.f, pzz = 0.f, pxy = 0.f, pxz = 0.f, pyz = 0.f;
    for (int i = 0; i < LBM_Q; i++) {
        float d = pop[(long long)i * N + n] - geq(i, rho, lx, ly, lz, usq);
        pxx += CX[i] * CX[i] * d;
        pyy += CY[i] * CY[i] * d;
        pzz += CZ[i] * CZ[i] * d;
        pxy += CX[i] * CY[i] * d;
        pxz += CX[i] * CZ[i] * d;
        pyz += CY[i] * CZ[i] * d;
    }
    // strain from Π^neq at the node's τ (molecular + sponge), stress with the molecular viscosity only: σ = 2ρν0 S =
    // −(τ0 − ½)/τ Π^neq_pre (the sponge's artificial viscosity inflated the outlet WSS ~190×)
    float tau = p.tau0 + spg[n];
    float keep = 1.f - 1.f / tau;
    if (fabsf(keep) < 0.05f) return;
    float s = -(p.tau0 - 0.5f) / tau / keep * p.stress_out;
    float tr = (pxx + pyy + pzz) / 3.f;
    float sxx = s * (pxx - tr), syy = s * (pyy - tr), szz = s * (pzz - tr), sxy = s * pxy, sxz = s * pxz, syz = s * pyz;
    float tx = sxx * nx + sxy * ny + sxz * nz, ty = sxy * nx + syy * ny + syz * nz, tz = sxz * nx + syz * ny + szz * nz;
    float tn = tx * nx + ty * ny + tz * nz;
    tx -= tn * nx;
    ty -= tn * ny;
    tz -= tn * nz;
    ATOMIC_ADD(a + 10, 1.f);
    ATOMIC_ADD(a + 11, tx);
    ATOMIC_ADD(a + 12, ty);
    ATOMIC_ADD(a + 13, tz);
    ATOMIC_ADD(a + 14, sqrtf(tx * tx + ty * ty + tz * tz));
}

// ── diagnostics: per cut Σ (u − u_cut)·n and the node count over the first interior node layer in front of the plane
// (−1 < s ≤ 0, within 1.5 radii: the flux the cut actually delivers / drains and the section, lattice units), the
// equilibrium slab's node count; then Σ(ρ − 1), fluid count, max |u|, refills
// diag layout: [3·ncuts] per cut (flux sum, layer count, slab count), then [Σ(ρ−1), count, max|u|, refills]

// diagnostics: per cut (4 floats) the flux through the interior layer −1 < s ≤ 0 relative to the cut's velocity, the
// layer's node count, the slab's node count, Σ(ρ − 1) over the layer; per valve probe (2 floats: upstream 2k,
// downstream 2k + 1) Σ(ρ − 1) and the node count; then Σ(ρ − 1), fluid count, max |u|, refills (mode 1: cuts and
// probes only — the inlet and valve feedback runs it every few dozen steps); with a region map, per region id (2 floats,
// before the totals) the fluid node count and Σ(ρ − 1)
// diag tail (after the 4 global slots): per cut Σ u_n s and Σ u_n over its flux layer (the outflow's spin history)
HD void node_diag(long long n, long long N, int ncuts, int nprobes, int nreg, int mode, const unsigned char* flags, const unsigned char* bcid, const unsigned char* pid, const unsigned char* rid, const float* cuts, const float* ru, const float* sc, float* diag) {
    unsigned char fl = flags[n];
    if (!(fl & F_FLUID)) return;
    float ux = ru[N + n], uy = ru[2 * N + n], uz = ru[3 * N + n];
    if (fl & F_BC) ATOMIC_ADD(diag + 4 * (bcid[n] - 1) + 2, 1.f);
    else if (bcid[n] >= 128 && bcid[n] - 128 < ncuts) {
        int c = bcid[n] - 128;
        const float* C = cuts + c * CUT_STRIDE;
        float un = (ux - C[16]) * C[3] + (uy - C[17]) * C[4] + (uz - C[18]) * C[5];
        ATOMIC_ADD(diag + 4 * c, un);
        ATOMIC_ADD(diag + 4 * c + 1, 1.f);
        ATOMIC_ADD(diag + 4 * c + 3, ru[n] - 1.f);
        float* ts = diag + 4 * ncuts + 2 * nprobes + 2 * nreg + 4 + 2 * c;
        ATOMIC_ADD(ts, un * sc[n]);
        ATOMIC_ADD(ts + 1, un);
    }
    if (pid[n] && pid[n] <= nprobes) {
        float* pr = diag + 4 * ncuts + 2 * (pid[n] - 1);
        ATOMIC_ADD(pr, ru[n] - 1.f);
        ATOMIC_ADD(pr + 1, 1.f);
    }
    if (mode) return;
    if (nreg && rid[n] && rid[n] < nreg) {
        float* rg = diag + 4 * ncuts + 2 * nprobes + 2 * rid[n];
        ATOMIC_ADD(rg, 1.f);
        ATOMIC_ADD(rg + 1, ru[n] - 1.f);
    }
    float* d = diag + 4 * ncuts + 2 * nprobes + 2 * nreg;
    ATOMIC_ADD(d + 0, ru[n] - 1.f);
    ATOMIC_ADD(d + 1, 1.f);
    float sp = sqrtf(ux * ux + uy * uy + uz * uz);
    if (!(sp == sp)) sp = 1e30f;  // NaN → flag as huge
    ATOMIC_MAXF(d + 2, sp);
    if (fl & F_REFILL) ATOMIC_ADD(d + 3, 1.f);
}
