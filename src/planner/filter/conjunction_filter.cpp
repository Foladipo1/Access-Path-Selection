#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/common/chrono.hpp"
namespace duckdb {

ConjunctionOrFilter::ConjunctionOrFilter() : ConjunctionFilter(TableFilterType::CONJUNCTION_OR) {
}

FilterPropagateResult ConjunctionOrFilter::CheckStatistics(BaseStatistics &stats) {
	// the OR filter is true if ANY of the children is true
	D_ASSERT(!child_filters.empty());
	for (auto &filter : child_filters) {
		auto prune_result = filter->CheckStatistics(stats);
		if (prune_result == FilterPropagateResult::NO_PRUNING_POSSIBLE) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else if (prune_result == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
	}
	return FilterPropagateResult::FILTER_ALWAYS_FALSE;
}

FilterPropagateResult ConjunctionOrFilter::CheckSketchStatistics(BaseStatistics &stats, idx_t index,
																 std::vector<std::shared_ptr<BaseColumnSketch>> &segment_sketches,
																 std::vector<ManagedSelection> &vector_sels) {
	// the OR filter is true if ANY of the children is true
	D_ASSERT(!child_filters.empty());
	for (auto &filter : child_filters) {
		auto prune_result = filter->CheckSketchStatistics(stats, index, segment_sketches, vector_sels);
		if (prune_result == FilterPropagateResult::NO_PRUNING_POSSIBLE) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else if (prune_result == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
	}
	return FilterPropagateResult::FILTER_ALWAYS_FALSE;
}

string ConjunctionOrFilter::ToString(const string &column_name) {
	string result;
	for (idx_t i = 0; i < child_filters.size(); i++) {
		if (i > 0) {
			result += " OR ";
		}
		result += child_filters[i]->ToString(column_name);
	}
	return result;
}

bool ConjunctionOrFilter::Equals(const TableFilter &other_p) const {
	if (!ConjunctionFilter::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<ConjunctionOrFilter>();
	if (other.child_filters.size() != child_filters.size()) {
		return false;
	}
	for (idx_t i = 0; i < other.child_filters.size(); i++) {
		if (!child_filters[i]->Equals(*other.child_filters[i])) {
			return false;
		}
	}
	return true;
}

ConjunctionAndFilter::ConjunctionAndFilter() : ConjunctionFilter(TableFilterType::CONJUNCTION_AND) {
}

FilterPropagateResult ConjunctionAndFilter::CheckStatistics(BaseStatistics &stats) {
	// the AND filter is true if ALL of the children is true
	D_ASSERT(!child_filters.empty());
	auto result = FilterPropagateResult::FILTER_ALWAYS_TRUE;
	for (auto &filter : child_filters) {
		auto prune_result = filter->CheckStatistics(stats);
		if (prune_result == FilterPropagateResult::FILTER_ALWAYS_FALSE) {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		} else if (prune_result != result) {
			result = FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
	}
	return result;
}

FilterPropagateResult ConjunctionAndFilter::CheckSketchStatistics(BaseStatistics &stats, idx_t index,
	 															  std::vector<std::shared_ptr<BaseColumnSketch>> &segment_sketches,
																  std::vector<ManagedSelection> &vector_sels) {
	// the AND filter is true if ALL of the children is true
	D_ASSERT(!child_filters.empty());
	auto result = FilterPropagateResult::FILTER_ALWAYS_TRUE;

	int i = 0;
	std::vector<uint64_t> merged_mask;																
	ManagedSelection &msel = vector_sels[index];

	for (auto &filter : child_filters) {

		auto prune_result = filter->CheckSketchStatistics(stats, index, segment_sketches, vector_sels);

		if (filter->filter_type == TableFilterType::CONSTANT_COMPARISON) {
			const auto &cur_mask = msel.bitmask;
			if(i == 0) {
				merged_mask = cur_mask;
			}
			else {
				for (size_t j = 0; j < merged_mask.size(); ++j) {
					merged_mask[j] &= cur_mask[j];
				}
			}
			i++;
		}
		
		if (prune_result == FilterPropagateResult::FILTER_ALWAYS_FALSE) {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		} else if (prune_result != result) {
			result = FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
	}

	if (--i) {
		msel.bitmask = merged_mask;
		msel.BitmaskToSelection();
	}

	return result;
}

string ConjunctionAndFilter::ToString(const string &column_name) {
	string result;
	for (idx_t i = 0; i < child_filters.size(); i++) {
		if (i > 0) {
			result += " AND ";
		}
		result += child_filters[i]->ToString(column_name);
	}
	return result;
}

bool ConjunctionAndFilter::Equals(const TableFilter &other_p) const {
	if (!ConjunctionFilter::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<ConjunctionAndFilter>();
	if (other.child_filters.size() != child_filters.size()) {
		return false;
	}
	for (idx_t i = 0; i < other.child_filters.size(); i++) {
		if (!child_filters[i]->Equals(*other.child_filters[i])) {
			return false;
		}
	}
	return true;
}

} // namespace duckdb
