#pragma once

#include <cudf/unary.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "types_shim.h"

namespace cudf_shims {

// ── Unary Operations ──────────────────────────────────────────

/// Apply a unary operation to a column.
/// `op` corresponds to cudf::unary_operator enum values.
std::unique_ptr<OwnedColumn> unary_operation(
    const OwnedColumn& input, int32_t op);

/// Return a bool8 column indicating which elements are null.
std::unique_ptr<OwnedColumn> is_null(const OwnedColumn& input);

/// Return a bool8 column indicating which elements are valid (non-null).
std::unique_ptr<OwnedColumn> is_valid(const OwnedColumn& input);

/// Return a bool8 column indicating which elements are NaN.
std::unique_ptr<OwnedColumn> is_nan(const OwnedColumn& input);

/// Return a bool8 column indicating which elements are not NaN.
std::unique_ptr<OwnedColumn> is_not_nan(const OwnedColumn& input);

/// Cast a column to a different data type.
std::unique_ptr<OwnedColumn> cast(const OwnedColumn& input, int32_t type_id);

/// Check if a cast between two data types is supported.
bool is_supported_cast(int32_t from_type_id, int32_t to_type_id);


/// Return a bool8 column indicating which elements are +/-infinity.
std::unique_ptr<OwnedColumn> is_inf(const OwnedColumn& input);

/// Return a bool8 column indicating which elements are NOT +/-infinity.
std::unique_ptr<OwnedColumn> is_not_inf(const OwnedColumn& input);

} // namespace cudf_shims
