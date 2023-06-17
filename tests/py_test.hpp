#pragma once

#define WITHOUT_NUMPY
#ifdef _DEBUG
#undef _DEBUG
#include <matplotlibcpp.h>
#define _DEBUG
#else
#include <matplotlibcpp.h>
#endif



std::vector<double> from_tensor(const at::Tensor& in)
{
	auto inf = in.contiguous().flatten();
	if (at::typeMetaToScalarType(inf.dtype()) == c10::ScalarType::Float) {
		std::vector<float> floatr(inf.data_ptr<float>(), inf.data_ptr<float>() + inf.numel());
		return std::vector<double>(floatr.begin(), floatr.end());
	}
	else if (at::typeMetaToScalarType(inf.dtype()) == c10::ScalarType::Double) {
		return std::vector<double>(inf.data_ptr<double>(), inf.data_ptr<double>() + inf.numel());
	}
	throw std::runtime_error("Unsupported dtype for from_flattened()");
}


void plot1_1(
	const at::Tensor& x1,
	int real_mag_imag = -1 /*real=-1, mag=0, imag=1*/)
{
	auto transf = [real_mag_imag](const at::Tensor& in) {
		if (real_mag_imag == -1)
			return from_tensor(at::real(in));
		else if (real_mag_imag == 0)
			return from_tensor(at::abs(in));
		else if (real_mag_imag == 1)
			return from_tensor(at::imag(in));
		else
			throw std::runtime_error("Unsupported real_mag_imag options for plot2");
	};

	namespace plt = matplotlibcpp;
	plt::figure();
	plt::plot(transf(x1), "r-*");
	plt::show();
}

void plot2_1(
	const at::Tensor& x1, const at::Tensor& x2,
	int real_mag_imag = -1 /*real=-1, mag=0, imag=1*/)
{
	auto transf = [real_mag_imag](const at::Tensor& in) {
		if (real_mag_imag == -1)
			return from_tensor(at::real(in));
		else if (real_mag_imag == 0)
			return from_tensor(at::abs(in));
		else if (real_mag_imag == 1)
			return from_tensor(at::imag(in));
		else
			throw std::runtime_error("Unsupported real_mag_imag options for plot2");
	};

	namespace plt = matplotlibcpp;
	plt::figure();
	plt::plot(transf(x1), "r-*");
	plt::plot(transf(x2), "g-*");
	plt::show();
}

void plot3_1(
	const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& x3,
	int real_mag_imag = -1 /*real=-1, mag=0, imag=1*/)
{
	auto transf = [real_mag_imag](const at::Tensor& in) {
		if (real_mag_imag == -1)
			return from_tensor(at::real(in));
		else if (real_mag_imag == 0)
			return from_tensor(at::abs(in));
		else if (real_mag_imag == 1)
			return from_tensor(at::imag(in));
		else
			throw std::runtime_error("Unsupported real_mag_imag options for plot2");
	};

	namespace plt = matplotlibcpp;
	plt::figure();
	plt::plot(transf(x1), "r-*");
	plt::plot(transf(x2), "g-*");
	plt::plot(transf(x3), "b-*");
	plt::show();
}



void plot1_2(
	const at::Tensor& x1, const at::Tensor& y1,
	int real_mag_imag = -1 /*real=-1, mag=0, imag=1*/)
{
	auto transf = [real_mag_imag](const at::Tensor& in) {
		if (real_mag_imag == -1)
			return from_tensor(at::real(in));
		else if (real_mag_imag == 0)
			return from_tensor(at::abs(in));
		else if (real_mag_imag == 1)
			return from_tensor(at::imag(in));
		else
			throw std::runtime_error("Unsupported real_mag_imag options for plot2");
	};

	namespace plt = matplotlibcpp;
	plt::figure();
	plt::plot(transf(x1), transf(y1), "r-*");
	plt::show();
}

void plot2_2(
	const at::Tensor& x1, const at::Tensor& y1, 
	const at::Tensor& x2, const at::Tensor& y2,
	int real_mag_imag = -1 /*real=-1, mag=0, imag=1*/)
{
	auto transf = [real_mag_imag](const at::Tensor& in) {
		if (real_mag_imag == -1)
			return from_tensor(at::real(in));
		else if (real_mag_imag == 0)
			return from_tensor(at::abs(in));
		else if (real_mag_imag == 1)
			return from_tensor(at::imag(in));
		else
			throw std::runtime_error("Unsupported real_mag_imag options for plot2");
	};

	namespace plt = matplotlibcpp;
	plt::figure();
	plt::plot(transf(x1), transf(y1), "r-*");
	plt::plot(transf(x2), transf(y2), "g-*");
	plt::show();
}

void plot3_2(
	const at::Tensor& x1, const at::Tensor& y1, 
	const at::Tensor& x2, const at::Tensor& y2,
	const at::Tensor& x3, const at::Tensor& y3,
	int real_mag_imag = -1 /*real=-1, mag=0, imag=1*/)
{
	auto transf = [real_mag_imag](const at::Tensor& in) {
		if (real_mag_imag == -1)
			return from_tensor(at::real(in));
		else if (real_mag_imag == 0)
			return from_tensor(at::abs(in));
		else if (real_mag_imag == 1)
			return from_tensor(at::imag(in));
		else
			throw std::runtime_error("Unsupported real_mag_imag options for plot2");
	};

	namespace plt = matplotlibcpp;
	plt::figure();
	plt::plot(transf(x1), transf(y1), "r-*");
	plt::plot(transf(x2), transf(y2), "g-*");
	plt::plot(transf(x3), transf(y3), "b-*");
	plt::show();
}

