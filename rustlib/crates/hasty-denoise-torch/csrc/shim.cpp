// Minimal C ABI over libtorch's C++ API for the hasty denoiser: load a TorchScript module (model.ts) or an
// AOTInductor package (model_aoti.pt2) and run forward(x, kv, h) on host buffers. Compiled against whatever libtorch
// the training venv provides (no generated bindings, so no libtorch version lock-in).
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <ATen/autocast_mode.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
struct Model {
  int kind;  // 0: TorchScript, 1: AOTInductor
  torch::Device device{torch::kCPU};
  std::unique_ptr<torch::jit::Module> ts;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> aoti;
  int autocast = 0;  // 0: float32, 1: bfloat16 autocast (CUDA, as in training), 2: float16 autocast
};

void set_err(char* err, int len, const std::string& msg) {
  if (err && len > 0) {
    std::strncpy(err, msg.c_str(), len - 1);
    err[len - 1] = 0;
  }
}
}  // namespace

extern "C" {

// device: −1 = CPU, ≥ 0 = CUDA index
void* hd_load(const char* path, int kind, int device, char* err, int errlen) {
  try {
    auto m = std::make_unique<Model>();
    m->kind = kind;
    m->device = device < 0 ? torch::Device(torch::kCPU) : torch::Device(torch::kCUDA, device);
    if (kind == 0) {
      m->ts = std::make_unique<torch::jit::Module>(torch::jit::load(path, m->device));
      m->ts->eval();
    } else {
      m->aoti = std::make_unique<torch::inductor::AOTIModelPackageLoader>(path, "model", false, 1, device < 0 ? -1 : device);
    }
    return m.release();
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return nullptr;
  }
}

int hd_cuda_available() { return torch::cuda::is_available() ? 1 : 0; }

// mixed precision for TorchScript models on CUDA (autocast, the training regime): 0 off, 1 bf16, 2 fp16
int hd_set_autocast(void* handle, int mode) {
  auto* m = static_cast<Model*>(handle);
  if (mode != 0 && (m->kind != 0 || !m->device.is_cuda())) return 1;
  m->autocast = mode;
  return 0;
}

// x: float32 [B, 7, 2, T, X, Y, Z] (shape in xshape[7]), kv: [B, 7], h: [B]; out: same size as x
int hd_forward(void* handle, const float* x, const int64_t* xshape, const float* kv, const float* h, float* out, char* err, int errlen) {
  try {
    auto* m = static_cast<Model*>(handle);
    c10::InferenceMode im;
    std::vector<int64_t> shp(xshape, xshape + 7);
    const int64_t B = shp[0];
    auto opt = torch::TensorOptions().dtype(torch::kFloat32);
    auto tx = torch::from_blob(const_cast<float*>(x), shp, opt).to(m->device);
    auto tk = torch::from_blob(const_cast<float*>(kv), {B, 7}, opt).to(m->device);
    auto th = torch::from_blob(const_cast<float*>(h), {B}, opt).to(m->device);
    torch::Tensor y;
    if (m->kind == 0) {
      if (m->autocast) {
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        at::autocast::set_autocast_dtype(at::kCUDA, m->autocast == 1 ? at::kBFloat16 : at::kHalf);
      }
      try {
        y = m->ts->forward({tx, tk, th}).toTensor();
      } catch (...) {
        if (m->autocast) at::autocast::set_autocast_enabled(at::kCUDA, false), at::autocast::clear_cache();
        throw;
      }
      if (m->autocast) at::autocast::set_autocast_enabled(at::kCUDA, false), at::autocast::clear_cache();
    } else {
      y = m->aoti->run({tx, tk, th})[0];
    }
    y = y.to(torch::kCPU).to(torch::kFloat32).contiguous();
    if (y.numel() != tx.numel()) {
      set_err(err, errlen, "output size differs from input size");
      return 1;
    }
    std::memcpy(out, y.data_ptr<float>(), sizeof(float) * y.numel());
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return 1;
  }
}

void hd_free(void* handle) { delete static_cast<Model*>(handle); }
}
