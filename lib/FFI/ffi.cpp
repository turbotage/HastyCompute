#include "ffi.hpp"

std::vector<c10::Stream> hasty::ffi::get_streams(const at::optional<std::vector<c10::Stream>>& streams)
{
	return hasty::torch_util::get_streams(streams);
}