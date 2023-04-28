
namespace hasty {

	template<int N>
	struct Block {
		std::array<int64_t, N> first_corner;
		std::array<int64_t, N> second_corner;

		Block() = default;

		static void randomize(Block& in, const std::array<int64_t, N>& bounds, const std::array<int64_t, N>& lens);

		static void emplace_back(std::vector<Block>& blocks, const std::array<int64_t, N>& bounds, const std::array<int64_t, N>& lens);

		static bool overlap(const Block& block1, const Block& block2);

		static bool overlap(const std::vector<Block>& blocks, const Block& block);

		static int overlap_idx(const std::vector<Block>& blocks, const Block& block);

		static const Block& overlap_ref(const std::vector<Block>& blocks, const Block& block);
	};



	template<int N>
	inline void Block<N>::randomize(Block& in, const std::array<int64_t, N>& bounds, const std::array<int64_t, N>& lens)
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

	template<int N>
	inline void Block<N>::emplace_back(std::vector<Block>& blocks, const std::array<int64_t, N>& bounds, const std::array<int64_t, N>& lens)
	{
		auto& last_block = blocks.emplace_back();
		while (true) {
			randomize(last_block, bounds, lens);
			if (!overlap(blocks, last_block))
				break;
		}
	}

	template<int N>
	inline bool Block<N>::overlap(const Block& block1, const Block& block2)
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

	template<int N>
	inline bool Block<N>::overlap(const std::vector<Block>& blocks, const Block& block)
	{
		for (const auto& blocki : blocks) {
			if (&blocki == &block)
				continue;
			if (overlap(blocki, block))
				return true;
		}
		return false;
	}

	template<int N>
	inline int Block<N>::overlap_idx(const std::vector<Block>& blocks, const Block& block)
	{
		for (int i = 0; i < blocks.size(); ++i) {
			if (Block::overlap(blocks[i], block))
				return i;
		}
		return blocks.size();
	}

	template<int N>
	inline const Block& Block<N>::overlap_ref(const std::vector<Block>& blocks, const Block& block)
	{
		for (const auto& blocki : blocks) {
			if (Block::overlap(blocki, block))
				return blocki;
		}
		return block;
	}

}