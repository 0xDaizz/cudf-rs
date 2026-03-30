#include "merge_shim.h"
#include "order_utils.h"
#include <cudf/merge.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <vector>

namespace cudf_shims {

std::unique_ptr<OwnedTable> merge_tables(
    const OwnedTable& left,
    const OwnedTable& right,
    rust::Slice<const int32_t> key_cols,
    rust::Slice<const int32_t> orders,
    rust::Slice<const int32_t> null_orders)
{
    std::vector<cudf::size_type> keys(key_cols.begin(), key_cols.end());
    auto col_order = to_column_order(orders);
    auto nul_order = to_null_order(null_orders);

    std::vector<cudf::table_view> tables_to_merge = {left.view(), right.view()};

    auto result = cudf::merge(
        tables_to_merge,
        keys,
        col_order,
        nul_order);

    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
