#pragma once

#include <array>
#include <vector>
#include <random>

namespace hasty {

	template<std::size_t N>
	struct Block {
		std::array<int64_t, N> first_corner;
		std::array<int64_t, N> second_corner;

		Block() = default;

		template<std::size_t N>
		static void randomize(Block<N>& in, const std::array<int64_t, N>& bounds, const std::array<int64_t, N>& lens)
		{
			static std::random_device dev;
			static std::mt19937 rng(dev());

			using int_dist = std::uniform_int_distribution<std::mt19937::result_type>;

			std::uniform_int_distribution<std::mt19937::result_type> dist;
			for (int i = 0; i < N; ++i) {
				dist = int_dist(0, bounds[i]);
				int64_t lower_bound = dist(rng);
				in.first_corner[i] = lower_bound;
				in.second_corner[i] = lower_bound + lens[i];
			}
		}

		template<std::size_t N>
		static void emplace_back(std::vector<Block<N>>& blocks, const std::array<int64_t, N>& bounds, const std::array<int64_t, N>& lens)
		{
			auto& last_block = blocks.emplace_back();
			while (true) {
				randomize(last_block, bounds, lens);
				if (!overlap(blocks, last_block))
					break;
			}
		}

		template<std::size_t N>
		static bool overlap(const Block<N>& block1, const Block<N>& block2)
		{
			for (int i = 0; i < N; ++i) {
				auto min1 = block1.first_corner[i];
				auto max1 = block1.second_corner[i];

				auto min2 = block2.first_corner[i];
				auto max2 = block2.second_corner[i];

				if ((min1 <= min2 && min2 <= max1) || (min1 <= max2 && max2 <= max1)) {
					return true;
				}
			}

			return false;
		}

		template<std::size_t N>
		static bool overlap(const std::vector<Block<N>>& blocks, const Block<N>& block)
		{
			for (const auto& blocki : blocks) {
				if (&blocki == &block)
					continue;
				if (overlap(blocki, block))
					return true;
			}
			return false;
		}

		template<std::size_t N>
		static int overlap_idx(const std::vector<Block<N>>& blocks, const Block<N>& block)
		{
			for (int i = 0; i < blocks.size(); ++i) {
				if (Block::overlap(blocks[i], block))
					return i;
			}
			return blocks.size();
		}

		template<std::size_t N>
		static const Block& overlap_ref(const std::vector<Block<N>>& blocks, const Block<N>& block)
		{
			for (const auto& blocki : blocks) {
				if (Block::overlap(blocki, block))
					return blocki;
			}
			return block;
		}


	};




}