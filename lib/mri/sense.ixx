module;

#include <torch/torch.h>

export module sense;

import <vector>;
import <functional>;

import torch_util;
import nufft;

namespace hasty {

	export using TensorVec = std::vector<at::Tensor>;

	namespace mri {

		export using CoilApplier = std::function<void(at::Tensor&, int32_t)>;

		export struct CoilManipulator {
			CoilManipulator() = default;

			CoilManipulator& setPreApply(const CoilApplier& apply) {
				if (preapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				preapplier = at::make_optional(apply);
				return *this;
			}

			CoilManipulator& setMidApply(const CoilApplier& apply) {
				if (midapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				midapplier = at::make_optional(apply);
				return *this;
			}

			CoilManipulator& setPostApply(const CoilApplier& apply) {
				if (postapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				postapplier = at::make_optional(apply);
				return *this;
			}

			at::optional<CoilApplier> preapplier;
			at::optional<CoilApplier> midapplier;
			at::optional<CoilApplier> postapplier;
		};

		class Sense;
		class CUDASense;

		export template<typename T>
			concept SenseConcept =
			std::same_as<T, CUDASense> ||
			std::same_as<T, Sense>;

		export class Sense {
		public:

			Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt)
				: _nufft(coords, nmodes, opts.has_value() ? *opts : fft::NufftOptions::type2()), _nmodes(nmodes)
			{
			}

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip)
			{
				sense_apply(*this, in, out, smaps, coils, imspace_storage, kspace_storage, premanip, postmanip);
			}

		private:
			fft::Nufft _nufft;
			std::vector<int64_t> _nmodes;

			template<SenseConcept T>
			friend void sense_apply(T& sense,
				const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip);

		};

		export class CUDASense {
		public:

			CUDASense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt)
				: _nufft(coords, nmodes, opts.has_value() ? *opts : fft::NufftOptions::type2()), _nmodes(nmodes)
			{
			}

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip)
			{
				sense_apply(*this, in, out, smaps, coils, imspace_storage, kspace_storage, premanip, postmanip);
			}

		private:
			fft::CUDANufft _nufft;
			std::vector<int64_t> _nmodes;

			template<SenseConcept T>
			friend void sense_apply(T& sense,
				const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip);

		};

		class SenseAdjoint;
		class CUDASenseAdjoint;

		export template<typename T>
			concept SenseAdjointConcept =
			std::same_as<T, CUDASenseAdjoint> ||
			std::same_as<T, SenseAdjoint>;

		export class SenseAdjoint {
		public:

			SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt)
				: _nufft(coords, nmodes, opts.has_value() ? *opts : fft::NufftOptions::type1()), _nmodes(nmodes)
			{
			}

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip)
			{
				sense_adjoint_apply(*this, in, out, smaps, coils, imspace_storage, kspace_storage, premanip, postmanip);
			}

		private:
			fft::Nufft _nufft;
			std::vector<int64_t> _nmodes;

			template<SenseAdjointConcept T>
			friend void sense_adjoint_apply(T& senseadjoint,
				const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip);
		};

		export class CUDASenseAdjoint {
		public:

			CUDASenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt)
				: _nufft(coords, nmodes, opts.has_value() ? *opts : fft::NufftOptions::type1()), _nmodes(nmodes)
			{
			}

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip)
			{
				sense_adjoint_apply(*this, in, out, smaps, coils, imspace_storage, kspace_storage, premanip, postmanip);
			}

		private:
			fft::CUDANufft _nufft;
			std::vector<int64_t> _nmodes;

			template<SenseAdjointConcept T>
			friend void sense_adjoint_apply(T& senseadjoint,
				const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip);

		};

		class SenseNormal;
		class CUDASenseNormal;

		export template<typename T>
		concept SenseNormalConcept =
			std::same_as<T, CUDASenseNormal> ||
			std::same_as<T, SenseNormal>;

		export class SenseNormal {
		public:

			SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<fft::NufftOptions>& backward_opts = at::nullopt)
				: _normal_nufft(coords, nmodes,
				forward_opts.has_value() ? *forward_opts : fft::NufftOptions::type2(),
				backward_opts.has_value() ? *backward_opts : fft::NufftOptions::type1()),
				_nmodes(nmodes)
			{
			}

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt,
				const at::optional<CoilApplier>& premanip = at::nullopt,
				const at::optional<CoilApplier>& midmanip = at::nullopt,
				const at::optional<CoilApplier>& postmanip = at::nullopt)
			{
				sense_normal_apply(*this, in, out, smaps, coils, imspace_storage, kspace_storage, premanip, midmanip, postmanip);
			}

			void apply_forward(const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt)
			{
				sense_normal_apply_forward(*this, in, out, smaps, coils, imspace_storage, kspace_storage);
			}

			void apply_backward(const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt)
			{
				sense_normal_apply_backward(*this, in, out, smaps, coils, kspace_storage, imspace_storage);
			}

		private:

			fft::NufftNormal _normal_nufft;
			std::vector<int64_t> _nmodes;

			template<SenseNormalConcept T>
			friend void sense_normal_apply(T& sensenormal,
				const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& midmanip,
				const at::optional<CoilApplier>& postmanip);

			template<SenseNormalConcept T>
			friend void sense_normal_apply_forward(T& sensenormal,
				const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage,
				const at::optional<at::Tensor>& kspace_storage);

			template<SenseNormalConcept T>
			friend void sense_normal_apply_backward(T& sensenormal,
				const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& kspace_storage,
				const at::optional<at::Tensor>& imspace_storage);

		};

		export class CUDASenseNormal {
		public:

			CUDASenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<fft::NufftOptions>& backward_opts = at::nullopt)
				: _normal_nufft(coords, nmodes,
				forward_opts.has_value() ? *forward_opts : fft::NufftOptions::type2(),
				backward_opts.has_value() ? *backward_opts : fft::NufftOptions::type1()),
				_nmodes(nmodes)
			{
			}

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt,
				const at::optional<CoilApplier>& premanip = at::nullopt,
				const at::optional<CoilApplier>& midmanip = at::nullopt,
				const at::optional<CoilApplier>& postmanip = at::nullopt)
			{
				sense_normal_apply(*this, in, out, smaps, coils, imspace_storage, kspace_storage, premanip, midmanip, postmanip);
			}

			void apply_forward(const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt)
			{
				sense_normal_apply_forward(*this, in, out, smaps, coils, imspace_storage, kspace_storage);
			}

			void apply_backward(const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt)
			{
				sense_normal_apply_backward(*this, in, out, smaps, coils, kspace_storage, imspace_storage);
			}

		private:

			fft::CUDANufftNormal _normal_nufft;
			std::vector<int64_t> _nmodes;

			template<SenseNormalConcept T>
			friend void sense_normal_apply(T& sensenormal,
				const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& midmanip,
				const at::optional<CoilApplier>& postmanip);

			template<SenseNormalConcept T>
			friend void sense_normal_apply_forward(T& sensenormal,
				const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage,
				const at::optional<at::Tensor>& kspace_storage);

			template<SenseNormalConcept T>
			friend void sense_normal_apply_backward(T& sensenormal,
				const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& kspace_storage,
				const at::optional<at::Tensor>& imspace_storage);

		};




















		template<SenseConcept T>
		void sense_apply(T& sense,
			const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps,
			const std::vector<int64_t>& coils, const at::optional<at::Tensor>& imspace_storage,
			const at::optional<at::Tensor>& kspace_storage, const at::optional<CoilApplier>& premanip,
			const at::optional<CoilApplier>& postmanip)
		{
			c10::InferenceMode inference_guard;

			if (coils.size() % sense._nmodes[0] != 0)
				throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

			int ntransf = sense._nmodes[0];
			int batch_runs = coils.size() / ntransf;

			out.zero_();

			at::Tensor imstore;
			if (imspace_storage.has_value()) {
				imstore = *imspace_storage;
			}
			else {
				imstore = at::empty(at::makeArrayRef(sense._nmodes), in.options());
			}

			at::Tensor kstore;
			if (kspace_storage.has_value()) {
				kstore = *kspace_storage;
			}
			else {
				kstore = at::empty({ ntransf, sense._nufft.nfreq() }, in.options());
			}

			bool accumulate = out.size(0) == 1;

			for (int brun = 0; brun < batch_runs; ++brun) {

				if (premanip.has_value()) {
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						imstore.select(0, j) = in.select(0, idx);
					}
					if (premanip.has_value()) {
						(*premanip)(imstore, brun);
					}
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						imstore.select(0, j).mul_(smaps.select(0, idx));
					}
				}
				else {
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						imstore.select(0, j) = in.select(0, 0) * smaps.select(0, idx);
					}
				}

				sense._nufft.apply(imstore, kstore);

				if (postmanip.has_value()) {
					(*postmanip)(kstore, brun);
				}

				if (accumulate) {
					for (int j = 0; j < ntransf; ++j) {
						out.add_(kstore.select(0, j));
					}
				}
				else {
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						out.select(0, idx).unsqueeze(0).add_(kstore.select(0, j));
					}
				}
			}
		}


		template<SenseAdjointConcept T>
		void sense_adjoint_apply(T& senseadjoint,
			const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps,
			const std::vector<int64_t>& coils, const at::optional<at::Tensor>& imspace_storage,
			const at::optional<at::Tensor>& kspace_storage, const at::optional<CoilApplier>& premanip,
			const at::optional<CoilApplier>& postmanip)
		{
			c10::InferenceMode inference_guard;

			if (coils.size() % senseadjoint._nmodes[0] != 0)
				throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

			int ntransf = senseadjoint._nmodes[0];
			int batch_runs = coils.size() / ntransf;

			out.zero_();

			at::Tensor imstore;
			if (imspace_storage.has_value()) {
				imstore = *imspace_storage;
			}
			else {
				imstore = at::empty(at::makeArrayRef(senseadjoint._nmodes), out.options());
			}

			at::Tensor kstore;
			if (kspace_storage.has_value()) {
				kstore = *kspace_storage;
			}
			else {
				kstore = at::empty({ ntransf, senseadjoint._nufft.nfreq() }, in.options());
			}

			bool accumulate = out.size(0) == 1;

			for (int brun = 0; brun < batch_runs; ++brun) {

				for (int j = 0; j < ntransf; ++j) {
					int idx = brun * ntransf + j;
					kstore.select(0, j) = in.select(0, idx);
				}

				if (premanip.has_value()) {
					(*premanip)(kstore, brun);
				}

				senseadjoint._nufft.apply(kstore, imstore);


				for (int j = 0; j < ntransf; ++j) {
					int idx = brun * ntransf + j;
					imstore.select(0, j).mul_(smaps.select(0, coils[idx]).conj());
				}

				if (postmanip.has_value()) {
					(*postmanip)(imstore, brun);
				}


				if (accumulate) {
					for (int j = 0; j < ntransf; ++j) {
						out.add_(imstore.select(0, j));
					}
				}
				else {
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						out.select(0, idx).add_(imstore.select(0, j));
					}
				}
			}
		}


		template<SenseNormalConcept T>
		void sense_normal_apply(T& sensenormal, const at::Tensor& in, at::Tensor& out,
			const at::Tensor& smaps, const std::vector<int64_t>& coils,
			const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
			const at::optional<CoilApplier>& premanip, const at::optional<CoilApplier>& midmanip,
			const at::optional<CoilApplier>& postmanip)
		{
			c10::InferenceMode inference_guard;

			if (coils.size() % sensenormal._nmodes[0] != 0)
				throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

			int ntransf = sensenormal._nmodes[0];
			int batch_runs = coils.size() / ntransf;

			out.zero_();

			at::Tensor imstore;
			if (imspace_storage.has_value()) {
				imstore = *imspace_storage;
			}
			else {
				imstore = at::empty(at::makeArrayRef(sensenormal._nmodes), in.options());
			}

			at::Tensor kstore;
			if (kspace_storage.has_value()) {
				kstore = *kspace_storage;
			}
			else {
				kstore = at::empty({ ntransf, sensenormal._normal_nufft.nfreq() }, in.options());
			}

			for (int brun = 0; brun < batch_runs; ++brun) {

				if (premanip.has_value()) {
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						imstore.select(0, j) = in.select(0, 0);
					}
					if (premanip.has_value()) {
						(*premanip)(imstore, brun);
					}
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						imstore.select(0, j).mul_(smaps.select(0, coils[idx]));
					}
				}
				else {
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						imstore.select(0, j) = in.select(0, 0) * smaps.select(0, coils[idx]);
					}
				}


				if (midmanip.has_value()) {
					sensenormal._normal_nufft.apply_inplace(imstore, kstore, std::bind(*midmanip, std::placeholders::_1, brun));
				}
				else {
					sensenormal._normal_nufft.apply_inplace(imstore, kstore, at::nullopt);
				}

				for (int j = 0; j < ntransf; ++j) {
					int idx = brun * ntransf + j;
					imstore.mul_(smaps.select(0, coils[idx]).conj());
				}

				if (postmanip.has_value()) {
					(*postmanip)(imstore, brun);
				}


				for (int j = 0; j < ntransf; ++j) {
					int idx = brun * ntransf + j;
					out.add_(imstore.select(0, j));
				}
			}
		}

		template<SenseNormalConcept T>
		void sense_normal_apply_forward(T& sensenormal,
			const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps,
			const std::vector<int64_t>& coils,
			const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage)
		{
			c10::InferenceMode inference_guard;

			if (coils.size() % sensenormal._nmodes[0] != 0)
				throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

			int ntransf = sensenormal._nmodes[0];
			int batch_runs = coils.size() / ntransf;

			out.zero_();

			at::Tensor imstore;
			if (imspace_storage.has_value()) {
				imstore = *imspace_storage;
			}
			else {
				imstore = at::empty(at::makeArrayRef(sensenormal._nmodes), in.options());
			}

			at::Tensor kstore;
			if (kspace_storage.has_value()) {
				kstore = *kspace_storage;
			}
			else {
				kstore = at::empty({ ntransf, sensenormal._normal_nufft.nfreq() }, in.options());
			}

			bool accumulate = out.size(0) == 1;

			for (int brun = 0; brun < batch_runs; ++brun) {

				for (int j = 0; j < ntransf; ++j) {
					int idx = brun * ntransf + j;
					imstore.select(0, j) = in.select(0, 0) * smaps.select(0, coils[idx]);
				}

				sensenormal._normal_nufft.apply_forward(imstore, kstore);

				for (int j = 0; j < ntransf; ++j) {
					int idx = brun * ntransf + j;
					out.select(0, idx).unsqueeze(0).add_(kstore.select(0, j));
				}
			}
		}

		template<SenseNormalConcept T>
		void sense_normal_apply_backward(T& sensenormal,
			const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps,
			const std::vector<int64_t>& coils, const at::optional<at::Tensor>& kspace_storage,
			const at::optional<at::Tensor>& imspace_storage)
		{
			c10::InferenceMode inference_guard;

			if (coils.size() % sensenormal._nmodes[0] != 0)
				throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

			int ntransf = sensenormal._nmodes[0];
			int batch_runs = coils.size() / ntransf;

			out.zero_();

			at::Tensor imstore;
			if (imspace_storage.has_value()) {
				imstore = *imspace_storage;
			}
			else {
				imstore = at::empty(at::makeArrayRef(sensenormal._nmodes), out.options());
			}

			at::Tensor kstore;
			if (kspace_storage.has_value()) {
				kstore = *kspace_storage;
			}
			else {
				kstore = at::empty({ ntransf, sensenormal._normal_nufft.nfreq() }, in.options());
			}

			bool accumulate = out.size(0) == 1;

			for (int brun = 0; brun < batch_runs; ++brun) {

				for (int j = 0; j < ntransf; ++j) {
					int idx = brun * ntransf + j;
					kstore.select(0, j) = in.select(0, idx);
				}

				sensenormal._normal_nufft.apply_backward(kstore, imstore);


				for (int j = 0; j < ntransf; ++j) {
					int idx = brun * ntransf + j;
					imstore.select(0, j).mul_(smaps.select(0, coils[idx]).conj());
				}


				if (accumulate) {
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						out.add_(imstore.select(0, j));
					}
				}
				else {
					for (int j = 0; j < ntransf; ++j) {
						int idx = brun * ntransf + j;
						out.select(0, idx).add_(imstore.select(0, j));
					}
				}
			}
		}




	}
}


