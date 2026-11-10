// CPU backend of the cardiac LBM: the per-node functions of lbm_core.h over node ranges [n0, n1). The Rust side splits
// the ranges over threads (tile-aligned, so per-tile accumulator cells never race) and gives each thread its own
// diagnostics buffer.
#include "lbm_core.h"

void cpu_motion(const MotionParams* mp, long long k0, long long k1, const unsigned short* Uall, float* Ut, float* Vt) {
    for (long long k = k0; k < k1; k++) node_motion(*mp, k, Uall, Ut, Vt);
}

void cpu_geom1(const GeomParams* p, long long n0, long long n1, long long N, const int* txyz, const float* Ut, const float* Vt,
               const float* sdf, const float* valves, const float* cuts, float* xref, float* phi, float* um, float* bsol,
               unsigned char* bcid, unsigned char* flags, float* spg, unsigned char* pid, const unsigned char* rmap, unsigned char* rid) {
    for (long long n = n0; n < n1; n++) node_geom1(*p, n, N, txyz, Ut, Vt, sdf, valves, cuts, xref, phi, um, bsol, bcid, flags, spg, pid, rmap, rid);
}

void cpu_geom2(long long n0, long long n1, long long N, const int* tnbr, const float* phi, unsigned int* links, unsigned char* qlink, unsigned char* flags) {
    for (long long n = n0; n < n1; n++) node_geom2(n, N, tnbr, phi, links, qlink, flags);
}

void cpu_refill(long long n0, long long n1, long long N, const int* tnbr, const unsigned char* flags, const float* um, float* f, float* ru) {
    for (long long n = n0; n < n1; n++) node_refill(n, N, tnbr, flags, um, f, ru);
}

void cpu_commit(long long n0, long long n1, unsigned char* flags) {
    for (long long n = n0; n < n1; n++) node_commit(n, flags);
}

void cpu_step(const StepParams* p, long long n0, long long n1, long long N, const int* tnbr, const unsigned char* flags,
              const unsigned int* links, const unsigned char* qlink, const float* um, const float* bsol, const unsigned char* bcid,
              const float* cuts, const float* src, float* dst, const float* ru_prev, float* ru_cur, const float* spg) {
    for (long long n = n0; n < n1; n++) node_step(*p, n, N, tnbr, flags, links, qlink, um, bsol, bcid, cuts, src, dst, ru_prev, ru_cur, spg);
}

void cpu_accum(const StepParams* p, long long n0, long long n1, long long N, const unsigned char* flags, const float* bsol, const float* ru, float* acc,
               const int* tnbr, const float* phi, const float* spg, const float* f, const float* sc) {
    for (long long n = n0; n < n1; n++) node_accum(*p, n, N, flags, bsol, ru, acc, tnbr, phi, spg, f, sc);
}

void cpu_scalar(const SpinParams* sp, int pass, long long n0, long long n1, long long N, const int* txyz, const int* tnbr, const unsigned char* flags, const unsigned char* bcid,
                const float* cuts, const float* ru, const float* atab, const float* s_old, const float* aux, float* s_new) {
    for (long long n = n0; n < n1; n++) node_scalar(*sp, pass, n, N, txyz, tnbr, flags, bcid, cuts, ru, atab, s_old, aux, s_new);
}

void cpu_srefill(long long n0, long long n1, const int* tnbr, const unsigned char* flags, float* s) {
    for (long long n = n0; n < n1; n++) node_srefill(n, tnbr, flags, s);
}

void cpu_diag(long long n0, long long n1, long long N, int ncuts, int nprobes, int nreg, int mode, const unsigned char* flags, const unsigned char* bcid, const unsigned char* pid, const unsigned char* rid, const float* cuts, const float* ru, const float* sc, float* diag) {
    for (long long n = n0; n < n1; n++) node_diag(n, N, ncuts, nprobes, nreg, mode, flags, bcid, pid, rid, cuts, ru, sc, diag);
}
