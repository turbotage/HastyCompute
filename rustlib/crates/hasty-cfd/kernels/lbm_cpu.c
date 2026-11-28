// CPU backend of the cardiac LBM: the per-node functions of lbm_core.h over node ranges [n0, n1). The Rust side splits
// the ranges over threads (tile-aligned, so per-tile accumulator cells never race) and gives each thread its own
// diagnostics buffer.
#include "lbm_core.h"

void cpu_motion(const MotionParams* mp, long long k0, long long k1, const unsigned short* Uall, float* Ut, float* Vt) {
    for (long long k = k0; k < k1; k++) node_motion(*mp, k, Uall, Ut, Vt);
}

void cpu_geom1(const GeomParams* p, long long n0, long long n1, long long N, const int* txyz, const float* Ut, const float* Vt,
               const float* sdf, const float* valves, const float* cuts, float* xref, float* phi, float* um, float* bsol,
               unsigned char* bcid, unsigned char* flags, float* spg, unsigned char* pid, const unsigned char* rmap, unsigned char* rid,
               const unsigned char* tgrp, const float* tw) {
    for (long long n = n0; n < n1; n++) node_geom1(*p, n, N, txyz, Ut, Vt, sdf, valves, cuts, xref, phi, um, bsol, bcid, flags, spg, pid, rmap, rid, tgrp, tw);
}

void cpu_geom2(const GeomParams* p, long long n0, long long n1, long long N, const int* txyz, const int* tnbr, const float* phi, const float* bsol,
               const float* valves, const float* xref, float* um, unsigned int* links, unsigned char* qlink, unsigned char* flags) {
    for (long long n = n0; n < n1; n++) node_geom2(*p, n, N, txyz, tnbr, phi, bsol, valves, xref, um, links, qlink, flags);
}

void cpu_refill(long long n0, long long n1, long long N, const int* tnbr, const unsigned char* flags, const float* um, float* f, float* ru) {
    for (long long n = n0; n < n1; n++) node_refill(n, N, tnbr, flags, um, f, ru);
}

void cpu_commit(long long n0, long long n1, unsigned char* flags) {
    for (long long n = n0; n < n1; n++) node_commit(n, flags);
}

void cpu_step(const StepParams* p, long long n0, long long n1, long long N, const int* tnbr, const unsigned char* flags,
              const unsigned int* links, const unsigned char* qlink, const float* um, const float* bsol, const unsigned char* bcid,
              const float* cuts, const float* src, float* dst, const float* ru_prev, float* ru_cur, const float* spg, float* dm, const unsigned char* wsc) {
    for (long long n = n0; n < n1; n++) node_step(*p, n, N, tnbr, flags, links, qlink, um, bsol, bcid, cuts, src, dst, ru_prev, ru_cur, spg, dm, wsc);
}

void cpu_accum(const StepParams* p, long long n0, long long n1, long long N, const unsigned char* flags, const float* bsol, const float* ru, float* acc,
               const int* tnbr, const float* phi, const float* spg, const float* f, const float* sc, const int* txyz, const int* atile) {
    for (long long n = n0; n < n1; n++) node_accum(*p, n, N, flags, bsol, ru, acc, tnbr, phi, spg, f, sc, txyz, atile);
}

void cpu_scalar(const SpinParams* sp, int pass, long long n0, long long n1, long long N, const int* txyz, const int* tnbr, const unsigned char* flags, const unsigned char* bcid,
                const float* cuts, const float* ru, const float* atab, const float* s_old, const float* aux, float* s_new) {
    for (long long n = n0; n < n1; n++) node_scalar(*sp, pass, n, N, txyz, tnbr, flags, bcid, cuts, ru, atab, s_old, aux, s_new);
}

void cpu_srefill(long long n0, long long n1, const int* tnbr, const unsigned char* flags, float* s) {
    for (long long n = n0; n < n1; n++) node_srefill(n, tnbr, flags, s);
}

void cpu_diag(long long n0, long long n1, long long N, int ncuts, int nprobes, int nreg, int mode, const unsigned char* flags, const unsigned char* bcid, const unsigned char* pid, const unsigned char* rid, const float* cuts, const float* ru, const float* sc, const int* tnbr, const float* f, int nvp, const float* vp, const float* xref, const unsigned int* links, float* diag) {
    for (long long n = n0; n < n1; n++) node_diag(n, N, ncuts, nprobes, nreg, mode, flags, bcid, pid, rid, cuts, ru, sc, tnbr, f, nvp, vp, xref, links, diag);
}

void cpu_wallmass(long long n0, long long n1, const unsigned char* flags, const float* ru, float* dm) {
    for (long long n = n0; n < n1; n++) node_wallmass(n, flags, ru, dm);
}

void cpu_wmreduce(long long n0, long long n1, const unsigned char* flags, const unsigned char* rid, unsigned int wmask, int keep, float* dm, float* sums) {
    for (long long n = n0; n < n1; n++) node_wmreduce(n, flags, rid, wmask, keep, dm, sums);
}

void cpu_wmapply(long long n0, long long n1, long long N, const unsigned char* flags, const unsigned char* rid, const float* corr, float* f, float* ru) {
    for (long long n = n0; n < n1; n++) node_wmapply(n, N, flags, rid, corr, f, ru);
}

// ── grid refinement (list entries [k0, k1)) ──

void cpu_explode(long long k0, long long k1, long long Nf, long long Nc, const int* glist, const int* gown, float rnear, float rfar, const float* fc, const float* ruc,
                 const float* scc, const int* tnbrc, const unsigned char* flc, const int* txyzf, const int* tnbrf, const unsigned char* flf,
                 float* fa, float* fb, float* rua, float* rub, float* sf) {
    for (long long g = k0; g < k1; g++) node_explode(g, Nf, Nc, glist, gown, rnear, rfar, fc, ruc, scc, tnbrc, flc, txyzf, tnbrf, flf, fa, fb, rua, rub, sf);
}

void cpu_ghostmask(long long k0, long long k1, const int* glist, const int* gown, const unsigned char* flc, unsigned char* flf) {
    for (long long g = k0; g < k1; g++) node_ghostmask(g, glist, gown, flc, flf);
}

void cpu_iflux(long long k0, long long k1, long long Nf, const int* glist, const int* tnbr, const unsigned char* flags, const unsigned int* links,
               const float* src, int first, float* gio) {
    for (long long g = k0; g < k1; g++) node_iflux(g, Nf, glist, tnbr, flags, links, src, first, gio);
}

void cpu_restrict(long long k0, long long k1, long long Nf, long long Nc, const int* hrim, const int* hsub, const unsigned char* flf, const float* ruf,
                  const float* sf, float* ruc, float* ruc2, float* sc) {
    for (long long h = k0; h < k1; h++) node_restrict(h, Nf, Nc, hrim, hsub, flf, ruf, sf, ruc, ruc2, sc);
}

void cpu_coalesce(long long k0, long long k1, long long Nf, long long Nc, const int* rim, const int* csub, const int* glist, const float* gio,
                  const int* tnbrc, const unsigned char* flc, const unsigned int* linksc, const float* ruc, const float* umc, float kfd,
                  const unsigned char* flf, const float* ff, float* fc) {
    for (long long r = k0; r < k1; r++) node_coalesce(r, Nf, Nc, rim, csub, glist, gio, tnbrc, flc, linksc, ruc, umc, kfd, flf, ff, fc);
}
