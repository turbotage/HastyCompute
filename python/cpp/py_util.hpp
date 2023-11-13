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


#include <torch/extension.h>
#include <torch/library.h>
#include <torch/script.h>
#include <torch/custom_class.h>

#include "../../lib/hasty.hpp"

