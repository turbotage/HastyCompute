// CUDA backend of the cardiac LBM: one thread per node (or motion voxel), the per-node functions of lbm_core.h
// (prepended by the Rust side before NVRTC compilation).

#define NODE_ID long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x

extern "C" __global__ void k_motion(MotionParams mp, const unsigned short* Uall, float* Ut, float* Vt) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < mp.M) node_motion(mp, k, Uall, Ut, Vt);
}

extern "C" __global__ void k_geom1(GeomParams p, long long N, const int* txyz, const float* Ut, const float* Vt, const float* sdf,
                                   const float* valves, const float* cuts, float* xref, float* phi, float* um, float* bsol,
                                   unsigned char* bcid, unsigned char* flags, float* spg, unsigned char* pid, const unsigned char* rmap, unsigned char* rid,
                                   const unsigned char* tgrp, const float* tw) {
    NODE_ID;
    if (n < N) node_geom1(p, n, N, txyz, Ut, Vt, sdf, valves, cuts, xref, phi, um, bsol, bcid, flags, spg, pid, rmap, rid, tgrp, tw);
}

extern "C" __global__ void k_geom2(GeomParams p, long long N, const int* txyz, const int* tnbr, const float* phi, const float* bsol,
                                   const float* valves, const float* xref, float* um, unsigned int* links, unsigned char* qlink, unsigned char* flags) {
    NODE_ID;
    if (n < N) node_geom2(p, n, N, txyz, tnbr, phi, bsol, valves, xref, um, links, qlink, flags);
}

extern "C" __global__ void k_refill(long long N, const int* tnbr, const unsigned char* flags, const float* um, float* f, float* ru) {
    NODE_ID;
    if (n < N) node_refill(n, N, tnbr, flags, um, f, ru);
}

extern "C" __global__ void k_commit(long long N, unsigned char* flags) {
    NODE_ID;
    if (n < N) node_commit(n, flags);
}

extern "C" __global__ void k_step(StepParams p, long long N, const int* tnbr, const unsigned char* flags, const unsigned int* links,
                                  const unsigned char* qlink, const float* um, const float* bsol, const unsigned char* bcid,
                                  const float* cuts, const float* src, float* dst, const float* ru_prev, float* ru_cur, const float* spg, float* dm, const unsigned char* wsc) {
    NODE_ID;
    if (n < N) node_step(p, n, N, tnbr, flags, links, qlink, um, bsol, bcid, cuts, src, dst, ru_prev, ru_cur, spg, dm, wsc);
}

extern "C" __global__ void k_accum(StepParams p, long long N, const unsigned char* flags, const float* bsol, const float* ru, float* acc,
                                   const int* tnbr, const float* phi, const float* spg, const float* f, const float* sc, const int* txyz, const int* atile) {
    NODE_ID;
    if (n < N) node_accum(p, n, N, flags, bsol, ru, acc, tnbr, phi, spg, f, sc, txyz, atile);
}

extern "C" __global__ void k_scalar(SpinParams sp, int pass, long long N, const int* txyz, const int* tnbr, const unsigned char* flags, const unsigned char* bcid,
                                    const float* cuts, const float* ru, const float* atab, const float* s_old, const float* aux, float* s_new) {
    NODE_ID;
    if (n < N) node_scalar(sp, pass, n, N, txyz, tnbr, flags, bcid, cuts, ru, atab, s_old, aux, s_new);
}

extern "C" __global__ void k_srefill(long long N, const int* tnbr, const unsigned char* flags, float* s) {
    NODE_ID;
    if (n < N) node_srefill(n, tnbr, flags, s);
}

extern "C" __global__ void k_diag(long long N, int ncuts, int nprobes, int nreg, int mode, const unsigned char* flags, const unsigned char* bcid, const unsigned char* pid, const unsigned char* rid, const float* cuts, const float* ru, const float* sc, const int* tnbr, const float* f, int nvp, const float* vp, const float* xref, const unsigned int* links, float* diag) {
    NODE_ID;
    if (n < N) node_diag(n, N, ncuts, nprobes, nreg, mode, flags, bcid, pid, rid, cuts, ru, sc, tnbr, f, nvp, vp, xref, links, diag);
}

extern "C" __global__ void k_wallmass(long long N, const unsigned char* flags, const float* ru, float* dm) {
    NODE_ID;
    if (n < N) node_wallmass(n, flags, ru, dm);
}

extern "C" __global__ void k_wmreduce(long long N, const unsigned char* flags, const unsigned char* rid, unsigned int wmask, int keep, float* dm, float* sums) {
    NODE_ID;
    if (n < N) node_wmreduce(n, flags, rid, wmask, keep, dm, sums);
}

extern "C" __global__ void k_wmapply(long long N, const unsigned char* flags, const unsigned char* rid, const float* corr, float* f, float* ru) {
    NODE_ID;
    if (n < N) node_wmapply(n, N, flags, rid, corr, f, ru);
}

// ── grid refinement (thread per list entry) ──

extern "C" __global__ void k_explode(long long ng, long long Nf, long long Nc, const int* glist, const int* gown, float rnear, float rfar, const float* fc, const float* ruc,
                                     const float* scc, const int* tnbrc, const unsigned char* flc, const int* txyzf, const int* tnbrf, const unsigned char* flf,
                                     float* fa, float* fb, float* rua, float* rub, float* sf) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < ng) node_explode(g, Nf, Nc, glist, gown, rnear, rfar, fc, ruc, scc, tnbrc, flc, txyzf, tnbrf, flf, fa, fb, rua, rub, sf);
}

extern "C" __global__ void k_ghostmask(long long ng, const int* glist, const int* gown, const unsigned char* flc, unsigned char* flf) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < ng) node_ghostmask(g, glist, gown, flc, flf);
}

extern "C" __global__ void k_iflux(long long ng, long long Nf, const int* glist, const int* tnbr, const unsigned char* flags, const unsigned int* links,
                                   const float* src, int first, float* gio) {
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < ng) node_iflux(g, Nf, glist, tnbr, flags, links, src, first, gio);
}

extern "C" __global__ void k_restrict(long long nh, long long Nf, long long Nc, const int* hrim, const int* hsub, const unsigned char* flf, const float* ruf,
                                      const float* sf, float* ruc, float* ruc2, float* sc) {
    long long h = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (h < nh) node_restrict(h, Nf, Nc, hrim, hsub, flf, ruf, sf, ruc, ruc2, sc);
}

extern "C" __global__ void k_coalesce(long long nr, long long Nf, long long Nc, const int* rim, const int* csub, const int* glist, const float* gio,
                                      const int* tnbrc, const unsigned char* flc, const unsigned int* linksc, const float* ruc, const float* umc, float kfd,
                                      const unsigned char* flf, const float* ff, float* fc) {
    long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < nr) node_coalesce(r, Nf, Nc, rim, csub, glist, gio, tnbrc, flc, linksc, ruc, umc, kfd, flf, ff, fc);
}
