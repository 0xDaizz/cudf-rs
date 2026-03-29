#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>
#include <memory>
#include "rust/cxx.h"
#include "types_shim.h"

namespace cudf_shims {

/// Owning wrapper around std::unique_ptr<cudf::column>.
///
/// This struct is exposed as an opaque type to Rust via cxx.
/// It owns GPU memory and frees it on destruction.
struct OwnedColumn {
    std::unique_ptr<cudf::column> inner;

    explicit OwnedColumn(std::unique_ptr<cudf::column> col)
        : inner(std::move(col)) {}

    // ── Accessors ──────────────────────────────────────────────
    int32_t size() const { return inner->size(); }
    int32_t type_id() const { return static_cast<int32_t>(inner->type().id()); }
    int32_t type_scale() const { return inner->type().scale(); }
    int32_t null_count() const { return inner->null_count(); }
    bool is_nullable() const { return inner->nullable(); }
    bool has_nulls() const { return inner->has_nulls(); }
    int32_t num_children() const { return inner->num_children(); }

    /// Get a non-owning view (for passing to libcudf functions).
    cudf::column_view view() const { return inner->view(); }
    cudf::mutable_column_view mutable_view() { return inner->mutable_view(); }
};

// ── Construction ───────────────────────────────────────────────

/// Create a column from host data by copying to GPU.
/// Template helper — instantiated for each supported type.
template <typename T>
std::unique_ptr<OwnedColumn> column_from_host(
    rust::Slice<const T> data, cudf::type_id tid);

// Concrete instantiations exposed to cxx:
std::unique_ptr<OwnedColumn> column_from_i8(rust::Slice<const int8_t> data);
std::unique_ptr<OwnedColumn> column_from_i16(rust::Slice<const int16_t> data);
std::unique_ptr<OwnedColumn> column_from_i32(rust::Slice<const int32_t> data);
std::unique_ptr<OwnedColumn> column_from_i64(rust::Slice<const int64_t> data);
std::unique_ptr<OwnedColumn> column_from_u8(rust::Slice<const uint8_t> data);
std::unique_ptr<OwnedColumn> column_from_u16(rust::Slice<const uint16_t> data);
std::unique_ptr<OwnedColumn> column_from_u32(rust::Slice<const uint32_t> data);
std::unique_ptr<OwnedColumn> column_from_u64(rust::Slice<const uint64_t> data);
std::unique_ptr<OwnedColumn> column_from_f32(rust::Slice<const float> data);
std::unique_ptr<OwnedColumn> column_from_f64(rust::Slice<const double> data);
std::unique_ptr<OwnedColumn> column_from_bool(rust::Slice<const bool> data);

std::unique_ptr<OwnedColumn> column_empty(int32_t type_id, int32_t size);

// ── Data Transfer ──────────────────────────────────────────────

/// Copy GPU column data to a host buffer.
void column_to_i32(const OwnedColumn& col, rust::Slice<int32_t> out);
void column_to_i64(const OwnedColumn& col, rust::Slice<int64_t> out);
void column_to_f32(const OwnedColumn& col, rust::Slice<float> out);
void column_to_f64(const OwnedColumn& col, rust::Slice<double> out);
void column_to_u8(const OwnedColumn& col, rust::Slice<uint8_t> out);
void column_null_mask(const OwnedColumn& col, rust::Slice<uint8_t> out);

} // namespace cudf_shims
