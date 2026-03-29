#pragma once

#include <cudf/binaryop.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "scalar_shim.h"
#include "types_shim.h"

namespace cudf_shims {

// ── Binary Operations ─────────────────────────────────────────

/// Binary operation: column op column.
/// `op` corresponds to cudf::binary_operator enum values.
/// `output_type` is the cudf::type_id of the result column.
std::unique_ptr<OwnedColumn> binary_operation_col_col(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs,
    int32_t op,
    int32_t output_type);

/// Binary operation: column op scalar.
std::unique_ptr<OwnedColumn> binary_operation_col_scalar(
    const OwnedColumn& lhs,
    const OwnedScalar& rhs,
    int32_t op,
    int32_t output_type);

/// Binary operation: scalar op column.
std::unique_ptr<OwnedColumn> binary_operation_scalar_col(
    const OwnedScalar& lhs,
    const OwnedColumn& rhs,
    int32_t op,
    int32_t output_type);

} // namespace cudf_shims
