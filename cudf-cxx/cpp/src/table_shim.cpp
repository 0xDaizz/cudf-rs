#include "table_shim.h"
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

// ── Construction ─────────────────────────────────────────────

std::unique_ptr<TableBuilder> table_builder_new() {
    return std::make_unique<TableBuilder>();
}

// ── Column access ────────────────────────────────────────────

std::unique_ptr<OwnedColumn> table_get_column(
    const OwnedTable& table, int32_t index)
{
    auto view = table.view();
    if (index < 0 || index >= view.num_columns()) {
        throw std::runtime_error("Column index out of bounds");
    }

    // Copy the column (table retains its data)
    auto col_view = view.column(index);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto copied = std::make_unique<cudf::column>(col_view, stream, mr);
    return std::make_unique<OwnedColumn>(std::move(copied));
}

std::unique_ptr<OwnedColumn> table_release_column(
    OwnedTable& table, int32_t index)
{
    table.ensure_released();
    auto& columns = table.released_columns;
    if (index < 0 || static_cast<size_t>(index) >= columns.size()) {
        throw std::runtime_error("Column index out of bounds in table_release_column");
    }
    if (!columns[index]) {
        throw std::runtime_error("Column already released at index");
    }
    auto col = std::move(columns[index]);
    return std::make_unique<OwnedColumn>(std::move(col));
}

} // namespace cudf_shims
