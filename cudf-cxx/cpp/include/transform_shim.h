#pragma once

#include <cudf/transform.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Replace NaN values with nulls in a floating-point column.
std::unique_ptr<OwnedColumn> nans_to_nulls(const OwnedColumn& col);

/// Convert a boolean column to a bitmask (host bytes).
rust::Vec<uint8_t> bools_to_mask(const OwnedColumn& col);

/// Factorize a table: returns (distinct_keys_table, encoded_indices_column).
/// The indices column maps each row to the corresponding key row.
std::unique_ptr<OwnedTable> encode_table(
    const OwnedTable& input,
    std::unique_ptr<OwnedColumn>& out_indices);

/// One-hot-encode a column against a set of categories.
/// Returns a table where each column corresponds to a category.
std::unique_ptr<OwnedTable> one_hot_encode(
    const OwnedColumn& input,
    const OwnedColumn& categories);

/// Convert a bitmask (host bytes) to a boolean column.
/// `begin` and `end` specify the bit range.
std::unique_ptr<OwnedColumn> mask_to_bools(
    rust::Slice<const uint8_t> mask_data,
    int32_t begin_bit,
    int32_t end_bit);

/// Compute per-row bit count for all columns in the table.
std::unique_ptr<OwnedColumn> row_bit_count(const OwnedTable& table);

} // namespace cudf_shims
