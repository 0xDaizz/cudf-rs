#include "concatenate_shim.h"
#include <cudf/concatenate.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> ColumnConcatBuilder::build() const {
    if (views.empty()) {
        throw std::runtime_error("Cannot concatenate zero columns");
    }
    auto result = cudf::concatenate(views);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> TableConcatBuilder::build() const {
    if (views.empty()) {
        throw std::runtime_error("Cannot concatenate zero tables");
    }
    auto result = cudf::concatenate(views);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<ColumnConcatBuilder> new_column_concat_builder() {
    return std::make_unique<ColumnConcatBuilder>();
}

std::unique_ptr<TableConcatBuilder> new_table_concat_builder() {
    return std::make_unique<TableConcatBuilder>();
}

} // namespace cudf_shims
