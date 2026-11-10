// CUDA backend of the cardiac LBM: one thread per node (or motion voxel), the per-node functions of lbm_core.h
// (prepended by the Rust side before NVRTC compilation).

#define NODE_ID long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x

extern "C" __global__ void k_motion(MotionParams mp, const unsigned short* Uall, float* Ut, float* Vt) {
    long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < mp.M) node_motion(mp, k, Uall, Ut, Vt);
}

extern "C" __global__ void k_geom1(GeomParams p, long long N, const int* txyz, const float* Ut, const float* Vt, const float* sdf,
                                   const float* valves, const float* cuts, float* xref, float* phi, float* um, float* bsol,
                                   unsigned char* bcid, unsigned char* flags, float* spg, unsigned char* pid, const unsigned char* rmap, unsigned char* rid) {
    NODE_ID;
    if (n < N) node_geom1(p, n, N, txyz, Ut, Vt, sdf, valves, cuts, xref, phi, um, bsol, bcid, flags, spg, pid, rmap, rid);
}

extern "C" __global__ void k_geom2(long long N, const int* tnbr, const float* phi, unsigned int* links, unsigned char* qlink, unsigned char* flags) {
    NODE_ID;
    if (n < N) node_geom2(n, N, tnbr, phi, links, qlink, flags);
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
                                  const float* cuts, const float* src, float* dst, const float* ru_prev, float* ru_cur, const float* spg) {
    NODE_ID;
    if (n < N) node_step(p, n, N, tnbr, flags, links, qlink, um, bsol, bcid, cuts, src, dst, ru_prev, ru_cur, spg);
}

extern "C" __global__ void k_accum(StepParams p, long long N, const unsigned char* flags, const float* bsol, const float* ru, float* acc,
                                   const int* tnbr, const float* phi, const float* spg, const float* f, const float* sc) {
    NODE_ID;
    if (n < N) node_accum(p, n, N, flags, bsol, ru, acc, tnbr, phi, spg, f, sc);
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

extern "C" __global__ void k_diag(long long N, int ncuts, int nprobes, int nreg, int mode, const unsigned char* flags, const unsigned char* bcid, const unsigned char* pid, const unsigned char* rid, const float* cuts, const float* ru, const float* sc, float* diag) {
    NODE_ID;
    if (n < N) node_diag(n, N, ncuts, nprobes, nreg, mode, flags, bcid, pid, rid, cuts, ru, sc, diag);
}
