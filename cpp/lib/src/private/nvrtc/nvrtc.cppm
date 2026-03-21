module;

#include "pch.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>

export module nvrtc;

import std;

namespace hasty {
    namespace nvrtc {

        export struct NVRTC_ModFunc {
			CUmodule module = nullptr;
			CUfunction function = nullptr;

            ~NVRTC_ModFunc() {
                if (module) {
                    cuModuleUnload(module);
                }
            }
		};

        export void compile_nvrtc_kernel(
            NVRTC_ModFunc& modFunc,
            const char* kernel_code, const char* kernel_name,
            int device_idx
        ) 
        {
            nvrtcProgram prog;
            nvrtcCreateProgram(&prog, kernel_code, kernel_name, 0, nullptr, nullptr);
            nvrtcResult compileResult = nvrtcCompileProgram(prog, 0, nullptr);
            if (compileResult != NVRTC_SUCCESS) {
                size_t logSize;
                nvrtcGetProgramLogSize(prog, &logSize);
                std::vector<char> log(logSize);
                nvrtcGetProgramLog(prog, log.data());
                throw std::runtime_error("NVRTC compile error: " + std::string(log.data()));
            }
            size_t ptxSize;
            nvrtcGetPTXSize(prog, &ptxSize);
            std::vector<char> ptx(ptxSize);
            nvrtcGetPTX(prog, ptx.data());
            nvrtcDestroyProgram(&prog);

            CUdevice cuDevice;
            cuDeviceGet(&cuDevice, device_idx);
            CUcontext cuContext;
            cuCtxGetCurrent(&cuContext);
            if (!cuContext) {
                throw std::runtime_error("No current CUDA context for device ");
            }
            cuModuleLoadData(&modFunc.module, ptx.data());
            cuModuleGetFunction(&modFunc.function, modFunc.module, kernel_name);
        }

    }
}