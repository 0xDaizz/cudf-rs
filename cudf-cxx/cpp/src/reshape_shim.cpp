#include "reshape_shim.h"
#include <cudf/reshape.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> interleave_columns(const OwnedTable& table) {
    auto result = cudf::interleave_columns(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> tile(const OwnedTable& table, int32_t count) {
    auto result = cudf::tile(table.view(), static_cast<cudf::size_type>(count));
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> byte_cast(
    const OwnedColumn& col, bool flip_endianness)
{
    auto endian = flip_endianness
        ? cudf::flip_endianness::YES
        : cudf::flip_endianness::NO;
    auto result = cudf::byte_cast(col.view(), endian);
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
