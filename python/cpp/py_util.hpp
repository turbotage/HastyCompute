#pragma once

#ifdef _DEBUG
#undef _DEBUG
#include <torch/extension.h>
#define _DEBUG
#else
#include <torch/extension.h>
#endif
#include <torch/library.h>

#include "../../lib/FFI/ffi.hpp"

