// VkFFT (CUDA backend, vendored in cpp/lib/external/vkFFT) vs cuFFT baseline. C2C f32 out-of-place, forward.
//   build: rustlib/bench/cpp/build.sh      run: ./vkfft_bench 1d:1024:16384 3d:320,320,320 ...
#define VKFFT_BACKEND 1
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvrtc.h>
#include <vkFFT.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static float median(std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  cuInit(0);
  CUdevice dev; cuDeviceGet(&dev, 0);
  CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, dev); cuCtxSetCurrent(ctx);
  int iters = 10;
  std::vector<std::string> specs;
  for (int i = 1; i < argc; ++i) specs.push_back(argv[i]);
  if (specs.empty()) {
    for (int n : {16, 64, 128, 256, 512, 1000, 1024, 2048, 384, 640, 768, 1536}) specs.push_back("1d:" + std::to_string(n) + ":" + std::to_string((1 << 24) / n));
    for (int d : {128, 192, 256, 320, 384, 400, 512, 640}) specs.push_back("3d:" + std::to_string(d) + "," + std::to_string(d) + "," + std::to_string(d));
    specs.push_back("3d:512,512,256"); specs.push_back("3d:1024,1024,512");
  }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  printf("%-24s %8s | VkFFT %8s %6s | cuFFT %8s %6s | VkFFT/cuFFT\n", "spec", "MiB", "ms", "GB/s", "ms", "GB/s");
  for (auto& spec : specs) {
    std::vector<uint64_t> dims; uint64_t batch = 1;
    size_t p = spec.find(':'); std::string kind = spec.substr(0, p), rest = spec.substr(p + 1);
    if (kind == "1d") { size_t q = rest.find(':'); dims = {(uint64_t)std::stoull(rest.substr(0, q))}; batch = std::stoull(rest.substr(q + 1)); }
    else { size_t s0 = 0; while (true) { size_t c = rest.find(',', s0); dims.push_back(std::stoull(rest.substr(s0, c == std::string::npos ? c : c - s0))); if (c == std::string::npos) break; s0 = c + 1; } }
    uint64_t total = batch; for (auto d : dims) total *= d;
    uint64_t bytes = total * sizeof(cufftComplex);
    size_t freeB, totB; cudaMemGetInfo(&freeB, &totB);
    if (bytes * 2 + (256u << 20) > freeB) { printf("%-24s skipped\n", spec.c_str()); continue; }
    cufftComplex *in, *out; cudaMalloc(&in, bytes); cudaMalloc(&out, bytes);
    { std::vector<cufftComplex> h(total); for (size_t i = 0; i < total; ++i) { h[i].x = (float)rand() / RAND_MAX - 0.5f; h[i].y = (float)rand() / RAND_MAX - 0.5f; } cudaMemcpy(in, h.data(), bytes, cudaMemcpyHostToDevice); }

    // ---- VkFFT
    VkFFTConfiguration cfg = {}; VkFFTApplication app = {};
    cfg.FFTdim = dims.size();
    // VkFFT size[0] is the fastest (contiguous) axis; our dims are slowest-first
    for (size_t i = 0; i < dims.size(); ++i) cfg.size[i] = dims[dims.size() - 1 - i];
    cfg.numberBatches = batch;
    cfg.device = &dev;
    uint64_t bsz = bytes; cfg.bufferSize = &bsz; cfg.inputBufferSize = &bsz; cfg.isInputFormatted = 1;
    cfg.coalescedMemory = 64;
    float tv = NAN;
    VkFFTResult r = initializeVkFFT(&app, cfg);
    if (r == VKFFT_SUCCESS) {
      VkFFTLaunchParams lp = {}; void* bp = out; void* ip = in; lp.buffer = &bp; lp.inputBuffer = &ip;
      std::vector<float> t;
      for (int i = 0; i < iters + 3; ++i) { cudaEventRecord(e0); VkFFTAppend(&app, -1, &lp); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (i >= 3) t.push_back(ms); }
      tv = median(t); deleteVkFFT(&app);
    }
    // ---- cuFFT
    cufftHandle plan;
    if (kind == "1d") { int n = (int)dims[0]; cufftPlanMany(&plan, 1, &n, nullptr, 1, n, nullptr, 1, n, CUFFT_C2C, (int)batch); }
    else { std::vector<int> n(dims.begin(), dims.end()); cufftPlanMany(&plan, (int)n.size(), n.data(), nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 1); }
    std::vector<float> t;
    for (int i = 0; i < iters + 3; ++i) { cudaEventRecord(e0); cufftExecC2C(plan, in, out, CUFFT_FORWARD); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (i >= 3) t.push_back(ms); }
    float tc = median(t); cufftDestroy(plan);
    auto gb = [&](float ms) { return 2.0 * bytes / (ms * 1e-3) / 1e9; };
    printf("%-24s %8llu | VkFFT %8.3f %6.0f | cuFFT %8.3f %6.0f | %5.2fx%s\n", spec.c_str(), (unsigned long long)(bytes >> 20), tv, gb(tv), tc, gb(tc), tc / tv, r == VKFFT_SUCCESS ? "" : "  (VkFFT init failed)");
    cudaFree(in); cudaFree(out);
  }
}
