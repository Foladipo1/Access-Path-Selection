//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/filter/null_filter.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/table_filter.hpp"
#include "duckdb/common/types/selection_vector.hpp"

namespace duckdb {

class IsNullFilter : public TableFilter {
public:
	static constexpr const TableFilterType TYPE = TableFilterType::IS_NULL;

public:
	IsNullFilter();

public:
	FilterPropagateResult CheckStatistics(BaseStatistics &stats) override;
	FilterPropagateResult CheckSketchStatistics(BaseStatistics &stats, idx_t index, std::vector<std::shared_ptr<BaseColumnSketch>> &segment_sketches,
												std::vector<ManagedSelection> &vector_sels) override;
	string ToString(const string &column_name) override;
	void Serialize(Serializer &serializer) const override;
	static unique_ptr<TableFilter> Deserialize(Deserializer &deserializer);
};

class IsNotNullFilter : public TableFilter {
public:
	static constexpr const TableFilterType TYPE = TableFilterType::IS_NOT_NULL;

public:
	IsNotNullFilter();

public:
	FilterPropagateResult CheckStatistics(BaseStatistics &stats) override;
	FilterPropagateResult CheckSketchStatistics(BaseStatistics &stats, idx_t index, std::vector<std::shared_ptr<BaseColumnSketch>> &segment_sketches,
												std::vector<ManagedSelection> &vector_sels) override;
	string ToString(const string &column_name) override;
	void Serialize(Serializer &serializer) const override;
	static unique_ptr<TableFilter> Deserialize(Deserializer &deserializer);
};

} // namespace duckdb
