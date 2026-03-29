#pragma once

#include <cudf/null_mask.hpp>
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

// ── Null Mask Utilities ───────────────────────────────────────

/// Opaque wrapper around rmm::device_buffer for null masks.
struct OwnedDeviceBuffer {
    rmm::device_buffer inner;

    explicit OwnedDeviceBuffer(rmm::device_buffer buf)
        : inner(std::move(buf)) {}

    /// Size of the buffer in bytes.
    int32_t size_bytes() const {
        return static_cast<int32_t>(inner.size());
    }
};

// ── Existing functions (Phase 1) ─────────────────────────────

/// Count the number of valid (non-null) elements in a column.
int32_t valid_count(const OwnedColumn& col);

/// Return a copy of the column with its null mask removed.
std::unique_ptr<OwnedColumn> set_all_valid(const OwnedColumn& col);

// ── New functions (Phase 2) ──────────────────────────────────

/// Create a null mask device buffer.
/// `state`: 0=UNALLOCATED, 1=UNINITIALIZED, 2=ALL_VALID, 3=ALL_NULL.
std::unique_ptr<OwnedDeviceBuffer> create_null_mask(
    int32_t size, int32_t state);

/// Count null values in a column.
int32_t null_count_column(const OwnedColumn& col);

/// Compute the number of bytes needed for a bitmask of given size.
int32_t bitmask_allocation_size(int32_t number_of_bits);

/// Copy a column's null mask to host.
void copy_null_mask_to_host(
    const OwnedColumn& col,
    rust::Slice<uint8_t> out);

/// Create a new column with a null mask set from host-side bytes.
std::unique_ptr<OwnedColumn> set_null_mask_from_host(
    const OwnedColumn& col,
    rust::Slice<const uint8_t> mask,
    int32_t null_count);

/// Set bits in a device bitmask. Operates on the bitmask of the given column.
/// Returns a new column with the modified bitmask.
std::unique_ptr<OwnedColumn> set_null_mask_range(
    const OwnedColumn& col,
    int32_t begin_bit,
    int32_t end_bit,
    bool valid);

/// Copy a column's bitmask to host bytes.
/// Returns empty vec if column has no null mask.
rust::Vec<uint8_t> copy_bitmask_to_host(const OwnedColumn& col);

/// Builder for collecting column views for bitmask AND/OR operations.
struct BitmaskBuilder {
    std::vector<cudf::column_view> views;

    void add_column(const OwnedColumn& col) {
        views.push_back(col.view());
    }

    int32_t num_columns() const {
        return static_cast<int32_t>(views.size());
    }
};

/// Result of a bitmask AND/OR operation.
struct BitmaskResult {
    rust::Vec<uint8_t> mask;
    int32_t null_count;

    rust::Vec<uint8_t> get_mask() const { return mask; }
    int32_t get_null_count() const { return null_count; }
};

std::unique_ptr<BitmaskBuilder> bitmask_builder_new();

/// Bitwise AND of null masks from multiple columns.
std::unique_ptr<BitmaskResult> bitmask_and(const BitmaskBuilder& builder);

/// Bitwise OR of null masks from multiple columns.
std::unique_ptr<BitmaskResult> bitmask_or(const BitmaskBuilder& builder);

} // namespace cudf_shims
