// cuFINUFFT (Release build, v2.5.1) baseline for the 3-D NUFFT benchmark. Same point generators as crates/hasty-bench/src/bin/nufft_bench.rs.
//   nufft_baseline [--tol t] [--ntransf n] [--method 0|1|2] kind:M:N [...]      kind = uniform | radial   N = modes per dim (cube)
#include <cuda_runtime.h>
#include <cufinufft.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Rng { uint64_t s; double next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; uint64_t v = s * 0x2545F4914F6CDD1DULL; return (double)(v >> 11) / (double)(1ULL << 52) - 1.0; } };

static void gen(const std::string& kind, size_t M, std::vector<float>& x, std::vector<float>& y, std::vector<float>& z) {
  Rng r{12345}; x.resize(M); y.resize(M); z.resize(M);
  const double PI = 3.14159265358979323846;
  if (kind == "uniform") { for (size_t j = 0; j < M; ++j) { x[j] = (float)(r.next() * PI * 0.999); y[j] = (float)(r.next() * PI * 0.999); z[j] = (float)(r.next() * PI * 0.999); } }
  else { // radial spokes through the centre, T samples per spoke
    size_t T = 512; size_t S = (M + T - 1) / T;
    for (size_t s = 0; s < S; ++s) {
      double zc = r.next(), ph = (r.next() + 1.0) * PI; double st = std::sqrt(1 - zc * zc);
      double nx = st * std::cos(ph), ny = st * std::sin(ph), nz = zc;
      for (size_t t = 0; t < T && s * T + t < M; ++t) { double rr = PI * 0.999 * (t + 0.5) / T; size_t j = s * T + t; x[j] = (float)(rr * nx); y[j] = (float)(rr * ny); z[j] = (float)(rr * nz); }
    }
  }
}

static float median(std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  double tol = 1e-6; int ntransf = 1, method = 0; std::vector<std::string> specs;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--tol") tol = atof(argv[++i]); else if (a == "--ntransf") ntransf = atoi(argv[++i]); else if (a == "--method") method = atoi(argv[++i]); else specs.push_back(a);
  }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  printf("cuFINUFFT release f32 tol=%g ntransf=%d gpu_method=%d\n", tol, ntransf, method);
  printf("%-22s | setpts ms | type1 ms/transf | type2 ms/transf\n", "kind:M:N");
  for (auto& sp : specs) {
    char kind[32]; long M; long N; sscanf(sp.c_str(), "%[^:]:%ld:%ld", kind, &M, &N);
    std::vector<float> hx, hy, hz; gen(kind, M, hx, hy, hz);
    float *dx, *dy, *dz; cudaMalloc(&dx, M * 4); cudaMalloc(&dy, M * 4); cudaMalloc(&dz, M * 4);
    cudaMemcpy(dx, hx.data(), M * 4, cudaMemcpyHostToDevice); cudaMemcpy(dy, hy.data(), M * 4, cudaMemcpyHostToDevice); cudaMemcpy(dz, hz.data(), M * 4, cudaMemcpyHostToDevice);
    size_t nm = (size_t)N * N * N;
    cuFloatComplex *c, *f; cudaMalloc(&c, (size_t)M * ntransf * 8); cudaMalloc(&f, nm * ntransf * 8);
    { std::vector<cuFloatComplex> h(std::max((size_t)M * ntransf, nm * ntransf)); for (auto& v : h) { v.x = (float)rand() / RAND_MAX - 0.5f; v.y = (float)rand() / RAND_MAX - 0.5f; }
      cudaMemcpy(c, h.data(), (size_t)M * ntransf * 8, cudaMemcpyHostToDevice); cudaMemcpy(f, h.data(), nm * ntransf * 8, cudaMemcpyHostToDevice); }
    float t_setpts = NAN, t1 = NAN, t2 = NAN;
    for (int type = 1; type <= 2; ++type) {
      cufinufft_opts opts; cufinufft_default_opts(&opts); if (method) opts.gpu_method = method;
      cufinufftf_plan plan; int64_t nmodes[3] = {N, N, N};
      int ier = cufinufftf_makeplan(type, 3, nmodes, type == 1 ? 1 : -1, ntransf, (float)tol, &plan, &opts);
      if (ier > 1) { printf("makeplan failed %d\n", ier); break; }
      cudaDeviceSynchronize(); cudaEventRecord(e0);
      cufinufftf_setpts(plan, M, dx, dy, dz, 0, nullptr, nullptr, nullptr);
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (type == 1) t_setpts = ms;
      std::vector<float> t;
      for (int i = 0; i < 8; ++i) { cudaEventRecord(e0); cufinufftf_execute(plan, c, f); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); if (i >= 3) t.push_back(ms); }
      (type == 1 ? t1 : t2) = median(t) / ntransf;
      cufinufftf_destroy(plan);
    }
    printf("%-22s | %9.2f | %15.3f | %15.3f\n", sp.c_str(), t_setpts, t1, t2);
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(c); cudaFree(f);
  }
}
