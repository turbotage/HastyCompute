module;


export module hasty_util_mod:idx;

import std;
import :alias;
import :meta;
import :typing;
import hasty_torch_wrapper;

namespace hasty {

// export struct NoneIdx {};

// export struct EllipsisIdx {};

// export struct SliceIdx {
//     Opt<i64> start;
//     Opt<i64> stop;
//     Opt<i64> step;

//     SliceIdx() = default;

//     template<std::integral I>
//     SliceIdx(I ival) : start(ival) {}

//     template<std::integral I1, std::integral I2>
//     SliceIdx(I1 ival1, I2 ival2) : start(ival1), stop(ival2) {}

//     SliceIdx(Opt<i64> start, Opt<i64> stop, Opt<i64> step)
//         : start(start), stop(stop), step(step) {}

// };

// export template<typename T>
// concept index_type =    std::is_same_v<T, NoneIdx>
//                     ||  std::is_same_v<T, EllipsisIdx>
//                     ||  std::is_same_v<T, SliceIdx>
//                     ||  std::is_integral_v<T>;

// export using TensorIdx = std::variant<NoneIdx, EllipsisIdx, SliceIdx, i64>;

// export template<std::size_t R, index_type... Idx>
// constexpr std::size_t get_slice_rank()
// {
//     int none = 0;
// 	int ints = 0;
// 	int ellipsis = 0;

// 	((std::is_same_v<Idx, NoneIdx> ? ++none : 
// 	std::is_integral_v<Idx> ? ++ints : 
// 	std::is_same_v<Idx, EllipsisIdx> ? ++ellipsis : 0), ...);

// 	return R - ints + none;
// }

// export template<std::size_t R, index_type... Idx>
// constexpr std::size_t get_slice_rank(std::tuple<Idx...> idxs)
// {
//     int none;
// 	int ints;
// 	int ellipsis;

// 	for_sequence<std::tuple_size_v<decltype(idxs)>>([&](auto i) constexpr {
// 		//if constexpr(std::is_same_v<decltype(idxs.template get<i>()), None>) {
// 		if constexpr(std::is_same_v<decltype(std::get<i>(idxs)), NoneIdx>) {
// 			++none;
// 		} 
// 		//else if constexpr(std::is_integral_v<decltype(idxs.template get<i>())>) {
// 		else if constexpr(std::is_integral_v<decltype(std::get<i>(idxs))>) {
// 			++ints;
// 		}
// 		/*
// 		else if constexpr(std::is_same_v<decltype(idxss.template get<i>()), Ellipsis>) {
// 			++ellipsis;
// 		} 
// 		*/
// 	});

// 	return R - ints + none;
// }

// export template<std::size_t R, index_type... Itx>
// constexpr std::size_t get_slice_rank(Itx... idxs)
// {
//     return get_slice_rank<R>(std::make_tuple(idxs...));
// }

// export template<index_type Idx>
// hat::indexing::TensorIndex torchidx(Idx idx)
// {
//     if constexpr(std::is_same_v<Idx, NoneIdx>) {
// 		return hat::indexing::None;
// 	} 
// 	else if constexpr(std::is_same_v<Idx, EllipsisIdx>) {
// 		return hat::indexing::Ellipsis;
// 	}
// 	else if constexpr(std::is_same_v<Idx, SliceIdx>) {
// 		return hat::indexing::Slice(
// 			torch_optional<hc10::SymInt>(idx.start),
// 			torch_optional<hc10::SymInt>(idx.end),
// 			torch_optional<hc10::SymInt>(idx.step));
// 	} else if constexpr(std::is_integral_v<Idx>) {
// 		return idx;
// 	} else {
// 		static_assert(always_false<Idx>, "Unsupported type for torchidx");
// 	}
// }


}