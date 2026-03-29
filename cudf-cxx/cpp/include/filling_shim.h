#pragma once

#include <cudf/filling.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"
#include "scalar_shim.h"

namespace cudf_shims {

// ── Filling ───────────────────────────────────────────────────

/// Fill a column in-place from `begin` to `end` with the given scalar value.
void fill_in_place(
    OwnedColumn& destination,
    int32_t begin,
    int32_t end,
    const OwnedScalar& value);

/// Fill a column out-of-place from `begin` to `end` with the given scalar value.
/// Returns a new column.
std::unique_ptr<OwnedColumn> fill(
    const OwnedColumn& input,
    int32_t begin,
    int32_t end,
    const OwnedScalar& value);

/// Repeat the rows of a table `count` times.
std::unique_ptr<OwnedTable> repeat_table(
    const OwnedTable& table,
    int32_t count);

/// Repeat the rows of a table, where each row is repeated by the
/// corresponding value in the `counts` column.
std::unique_ptr<OwnedTable> repeat_table_variable(
    const OwnedTable& table,
    const OwnedColumn& counts);

/// Generate a sequence of values: init, init+step, init+2*step, ...
/// Uses i32 scalars.
std::unique_ptr<OwnedColumn> sequence_i32(
    int32_t size,
    int32_t init,
    int32_t step);

/// Generate a sequence of values using i64 scalars.
std::unique_ptr<OwnedColumn> sequence_i64(
    int32_t size,
    int64_t init,
    int64_t step);

/// Generate a sequence of values using f32 scalars.
std::unique_ptr<OwnedColumn> sequence_f32(
    int32_t size,
    float init,
    float step);

/// Generate a sequence of values using f64 scalars.
std::unique_ptr<OwnedColumn> sequence_f64(
    int32_t size,
    double init,
    double step);

} // namespace cudf_shims
