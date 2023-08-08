#pragma once

#if defined(_WIN32)
#define LIB_EXPORT __declspec(dllexport)
#define LIB_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#define LIB_EXPORT __attribute_((visibility("default")))
#define LIB_IMPORT
#else
#define LIB_EXPORT
#define LIB_IMPORT
#pragma warning Unknown dynamic link import/export semantics
#endif

/*
#ifdef _DEBUG
#undef _DEBUG
#include <torch/extension.h>
#define _DEBUG
#else
#include <torch/extension.h>
#endif
*/

#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <pybind11/functional.h>


#include "../../lib/hasty.hpp"
