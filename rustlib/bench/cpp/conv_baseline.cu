// Baselines for the zero-padded convolution (Toeplitz multiplication) pass, matching cpp/lib/src/fft/toeplitz_mult.cpp semantics:
//   out += conj-mult( crop( IFFT( K . FFT( pad( x . d1 ) ) ) ), d2 ) / prod(pad)
//  A) "VkFFT": load kernel (N^3 -> scratch) + VkFFT performConvolution with zero-padding (as the HastyCompute code) + store kernel (crop, post, accumulate)
//  B) "cuFFT naive": memset scratch + load + cufftExec C2C fwd + pointwise K + cufftExec inv + store
// usage: conv_baseline Zn,Yn,Xn [more ...]   (pad = 2x)
#define VKFFT_BACKEND 1
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvrtc.h>
#include <vkFFT.h>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

typedef cufftComplex cx;
__device__ __forceinline__ cx cmul(cx a, cx b) { return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
__device__ __forceinline__ cx cmulc(cx a, cx b) { return make_float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y); }

__global__ void load_k(cx* scratch, const cx* x, const cx* d1, long Zn, long Yn, long Xn, long Y2, long X2) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i >= Zn * Yn * Xn) return;
  long xx = i % Xn, yy = (i / Xn) % Yn, zz = i / (Xn * Yn);
  scratch[(zz * Y2 + yy) * X2 + xx] = cmul(x[i], d1[i]);
}
__global__ void store_k(cx* out, const cx* scratch, const cx* d2, long Zn, long Yn, long Xn, long Y2, long X2, float sc) {
  long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i >= Zn * Yn * Xn) return;
  long xx = i % Xn, yy = (i / Xn) % Yn, zz = i / (Xn * Yn);
  cx v = scratch[(zz * Y2 + yy) * X2 + xx]; v.x *= sc; v.y *= sc; v = cmulc(v, d2[i]);
  out[i].x += v.x; out[i].y += v.y;
}
__global__ void mulk(cx* a, const cx* k, long n) { long i = blockIdx.x * (long)blockDim.x + threadIdx.x; if (i < n) a[i] = cmul(a[i], k[i]); }

static float median(std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  cuInit(0); CUdevice dev; cuDeviceGet(&dev, 0); CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, dev); cuCtxSetCurrent(ctx);
  std::vector<std::string> specs; for (int i = 1; i < argc; ++i) specs.push_back(argv[i]);
  if (specs.empty()) specs = {"64,64,64", "128,128,128", "160,160,160", "192,192,192", "256,256,256", "320,320,320", "512,512,256", "512,512,320"};
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  printf("%-16s %10s | VkFFT-conv path ms | cuFFT naive ms\n", "N (pad=2N)", "pad MiB");
  for (auto& sp : specs) {
    long Zn, Yn, Xn; sscanf(sp.c_str(), "%ld,%ld,%ld", &Zn, &Yn, &Xn);
    long Z2 = 2 * Zn, Y2 = 2 * Yn, X2 = 2 * Xn; long nin = Zn * Yn * Xn, npad = Z2 * Y2 * X2;
    size_t freeB, totB; cudaMemGetInfo(&freeB, &totB);
    size_t need = (size_t)npad * 8 * 2 + (size_t)nin * 8 * 4 + (300u << 20);
    if (need > freeB) { printf("%-16s skipped\n", sp.c_str()); continue; }
    cx *x, *d1, *d2, *out, *scratch, *K; cudaMalloc(&x, nin * 8); cudaMalloc(&d1, nin * 8); cudaMalloc(&d2, nin * 8); cudaMalloc(&out, nin * 8); cudaMalloc(&scratch, npad * 8); cudaMalloc(&K, npad * 8);
    { std::vector<cx> h(std::max(nin, npad)); for (auto& c : h) { c.x = (float)rand() / RAND_MAX - 0.5f; c.y = (float)rand() / RAND_MAX - 0.5f; }
      cudaMemcpy(x, h.data(), nin * 8, cudaMemcpyHostToDevice); cudaMemcpy(d1, h.data(), nin * 8, cudaMemcpyHostToDevice); cudaMemcpy(d2, h.data(), nin * 8, cudaMemcpyHostToDevice);
      cudaMemcpy(K, h.data(), npad * 8, cudaMemcpyHostToDevice); cudaMemset(out, 0, nin * 8); cudaMemset(scratch, 0, npad * 8); }
    int bs = 256, g = (nin + bs - 1) / bs; float sc = 1.0f / npad;

    // ---- A) VkFFT convolution with zero padding
    VkFFTConfiguration cfg = {}; VkFFTApplication app = {};
    cfg.FFTdim = 3; cfg.size[0] = X2; cfg.size[1] = Y2; cfg.size[2] = Z2; cfg.device = &dev;
    uint64_t bsz = (uint64_t)npad * 8; cfg.bufferSize = &bsz; cfg.kernelSize = &bsz;
    cfg.performConvolution = 1; cfg.coalescedMemory = 64;
    uint64_t Ns[3] = {(uint64_t)Xn, (uint64_t)Yn, (uint64_t)Zn};
    for (int i = 0; i < 3; ++i) { cfg.performZeropadding[i] = 1; cfg.fft_zeropad_left[i] = Ns[i]; cfg.fft_zeropad_right[i] = 2 * Ns[i]; }
    float ta = NAN;
    VkFFTResult r = initializeVkFFT(&app, cfg);
    if (r == VKFFT_SUCCESS) {
      VkFFTLaunchParams lp = {}; void* bp = scratch; void* kp = K; lp.buffer = &bp; lp.kernel = &kp;
      std::vector<float> t;
      for (int i = 0; i < 8; ++i) {
        cudaEventRecord(e0);
        load_k<<<g, bs>>>(scratch, x, d1, Zn, Yn, Xn, Y2, X2);
        VkFFTAppend(&app, -1, &lp);
        store_k<<<g, bs>>>(out, scratch, d2, Zn, Yn, Xn, Y2, X2, sc);
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (i >= 3) t.push_back(ms);
      }
      ta = median(t); deleteVkFFT(&app);
    }
    // ---- B) cuFFT naive
    cufftHandle plan; int n3[3] = {(int)Z2, (int)Y2, (int)X2}; cufftPlan3d(&plan, n3[0], n3[1], n3[2], CUFFT_C2C);
    std::vector<float> t;
    for (int i = 0; i < 8; ++i) {
      cudaEventRecord(e0);
      cudaMemsetAsync(scratch, 0, (size_t)npad * 8);
      load_k<<<g, bs>>>(scratch, x, d1, Zn, Yn, Xn, Y2, X2);
      cufftExecC2C(plan, scratch, scratch, CUFFT_FORWARD);
      mulk<<<(npad + bs - 1) / bs, bs>>>(scratch, K, npad);
      cufftExecC2C(plan, scratch, scratch, CUFFT_INVERSE);
      store_k<<<g, bs>>>(out, scratch, d2, Zn, Yn, Xn, Y2, X2, sc);
      cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (i >= 3) t.push_back(ms);
    }
    float tb = median(t); cufftDestroy(plan);
    printf("%-16s %10ld | %18.3f%s | %14.3f\n", sp.c_str(), (long)(npad * 8 >> 20), ta, r == VKFFT_SUCCESS ? "" : " (init failed)", tb);
    cudaFree(x); cudaFree(d1); cudaFree(d2); cudaFree(out); cudaFree(scratch); cudaFree(K);
  }
}
