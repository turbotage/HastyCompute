module;

#include <torch/torch.h>
#include <highfive/H5Easy.hpp>

module hdf5;

import torch_util;
import hasty_util;

HighFive::CompoundType make_complex_float() {
	return {
		{"r", HighFive::AtomicType<float>{}},
		{"i", HighFive::AtomicType<float>{}}
	};
}

HIGHFIVE_REGISTER_TYPE(std::complex<float>, make_complex_float);

HighFive::CompoundType make_complex_double() {
	return {
		{"r", HighFive::AtomicType<double>{}},
		{"i", HighFive::AtomicType<double>{}}
	};
}

HIGHFIVE_REGISTER_TYPE(std::complex<double>, make_complex_double);


at::Tensor hasty::import_tensor(const std::string& filepath, const std::string& dataset)
{
	HighFive::File file(filepath, HighFive::File::ReadOnly);
	HighFive::DataSet dset = file.getDataSet(dataset);

	HighFive::DataType dtype = dset.getDataType();
	std::string dtype_str = dtype.string();
	size_t dtype_size = dtype.getSize();
	std::vector<int64_t> dims = hasty::util::vector_cast<int64_t>(dset.getDimensions());
	size_t nelem = dset.getElementCount();

	if (dtype_str == "Float32") {
		std::vector<float> data(nelem);
		dset.read(data.data());
		return at::from_blob(data.data(), at::makeArrayRef(dims), at::ScalarType::Float).detach().clone();
	}
	else if (dtype_str == "Float64") {
		std::vector<double> data(nelem);
		dset.read(data.data());
		return at::from_blob(data.data(), at::makeArrayRef(dims), at::ScalarType::Double).detach().clone();
	}
	else if (dtype_str == "Compound64") {
		HighFive::CompoundType ctype(std::move(dtype));
		auto members = ctype.getMembers();
		if (members.size() != 2)
			throw std::runtime_error("HighFive reported an Compound64 type");
		std::vector<std::complex<float>> data(nelem);
		dset.read(data.data());
		return at::from_blob(data.data(), at::makeArrayRef(dims), at::ScalarType::ComplexFloat).detach().clone();
	}
	else if (dtype_str == "Compound128") {
		HighFive::CompoundType ctype(std::move(dtype));
		auto members = ctype.getMembers();
		if (members.size() != 2)
			throw std::runtime_error("HighFive reported an Compound64 type");
		std::vector<std::complex<double>> data(nelem);
		dset.read(data.data());
		return at::from_blob(data.data(), at::makeArrayRef(dims), at::ScalarType::ComplexDouble).detach().clone();
	}
	else {
		throw std::runtime_error("disallowed dtype");
	}

}


