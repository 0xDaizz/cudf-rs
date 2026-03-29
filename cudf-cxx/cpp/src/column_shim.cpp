#include "column_shim.h"
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace cudf_shims {

// ── Template helper ────────────────────────────────────────────

template <typename T>
std::unique_ptr<OwnedColumn> column_from_host(
    rust::Slice<const T> data, cudf::type_id tid)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Copy host data to device buffer
    auto size = static_cast<cudf::size_type>(data.size());
    auto byte_size = size * sizeof(T);

    rmm::device_buffer dev_buf(data.data(), byte_size, stream, mr);

    auto col = std::make_unique<cudf::column>(
        cudf::data_type{tid},
        size,
        std::move(dev_buf),
        rmm::device_buffer{},  // no null mask
        0                      // null count = 0
    );

    return std::make_unique<OwnedColumn>(std::move(col));
}

// ── Explicit instantiations ────────────────────────────────────

std::unique_ptr<OwnedColumn> column_from_i8(rust::Slice<const int8_t> data) {
    return column_from_host(data, cudf::type_id::INT8);
}

std::unique_ptr<OwnedColumn> column_from_i16(rust::Slice<const int16_t> data) {
    return column_from_host(data, cudf::type_id::INT16);
}

std::unique_ptr<OwnedColumn> column_from_i32(rust::Slice<const int32_t> data) {
    return column_from_host(data, cudf::type_id::INT32);
}

std::unique_ptr<OwnedColumn> column_from_i64(rust::Slice<const int64_t> data) {
    return column_from_host(data, cudf::type_id::INT64);
}

std::unique_ptr<OwnedColumn> column_from_u8(rust::Slice<const uint8_t> data) {
    return column_from_host(data, cudf::type_id::UINT8);
}

std::unique_ptr<OwnedColumn> column_from_u16(rust::Slice<const uint16_t> data) {
    return column_from_host(data, cudf::type_id::UINT16);
}

std::unique_ptr<OwnedColumn> column_from_u32(rust::Slice<const uint32_t> data) {
    return column_from_host(data, cudf::type_id::UINT32);
}

std::unique_ptr<OwnedColumn> column_from_u64(rust::Slice<const uint64_t> data) {
    return column_from_host(data, cudf::type_id::UINT64);
}

std::unique_ptr<OwnedColumn> column_from_f32(rust::Slice<const float> data) {
    return column_from_host(data, cudf::type_id::FLOAT32);
}

std::unique_ptr<OwnedColumn> column_from_f64(rust::Slice<const double> data) {
    return column_from_host(data, cudf::type_id::FLOAT64);
}

std::unique_ptr<OwnedColumn> column_from_bool(rust::Slice<const bool> data) {
    // Bool requires special handling: cudf stores BOOL8 as int8
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto size = static_cast<cudf::size_type>(data.size());

    // Convert bool -> int8_t
    std::vector<int8_t> int_data(size);
    for (cudf::size_type i = 0; i < size; ++i) {
        int_data[i] = data[i] ? 1 : 0;
    }

    rmm::device_buffer dev_buf(int_data.data(), size * sizeof(int8_t), stream, mr);

    auto col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::BOOL8},
        size,
        std::move(dev_buf),
        rmm::device_buffer{},
        0
    );

    return std::make_unique<OwnedColumn>(std::move(col));
}

std::unique_ptr<OwnedColumn> column_empty(int32_t type_id, int32_t size) {
    auto tid = static_cast<cudf::type_id>(type_id);
    auto col = cudf::make_numeric_column(
        cudf::data_type{tid},
        size,
        cudf::mask_state::ALL_NULL,
        cudf::get_default_stream(),
        cudf::get_current_device_resource_ref()
    );
    return std::make_unique<OwnedColumn>(std::move(col));
}

// ── Data Transfer (GPU → Host) ─────────────────────────────────

template <typename T>
void column_to_host(const OwnedColumn& col, rust::Slice<T> out) {
    auto view = col.view();
    if (static_cast<size_t>(view.size()) > out.size()) {
        throw std::runtime_error("Output buffer too small");
    }

    auto stream = cudf::get_default_stream();
    cudaMemcpyAsync(
        out.data(),
        view.data<T>(),
        view.size() * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream.value()
    );
    stream.synchronize();
}

void column_to_i32(const OwnedColumn& col, rust::Slice<int32_t> out) {
    column_to_host(col, out);
}

void column_to_i64(const OwnedColumn& col, rust::Slice<int64_t> out) {
    column_to_host(col, out);
}

void column_to_f32(const OwnedColumn& col, rust::Slice<float> out) {
    column_to_host(col, out);
}

void column_to_f64(const OwnedColumn& col, rust::Slice<double> out) {
    column_to_host(col, out);
}

void column_to_u8(const OwnedColumn& col, rust::Slice<uint8_t> out) {
    column_to_host(col, out);
}

void column_null_mask(const OwnedColumn& col, rust::Slice<uint8_t> out) {
    auto view = col.view();
    if (!view.nullable()) {
        // Fill with 0xFF (all valid)
        std::memset(out.data(), 0xFF, out.size());
        return;
    }

    auto mask = view.null_mask();
    auto num_bytes = cudf::bitmask_allocation_size_bytes(view.size());
    if (num_bytes > out.size()) {
        throw std::runtime_error("Output buffer too small for null mask");
    }

    auto stream = cudf::get_default_stream();
    cudaMemcpyAsync(
        out.data(),
        mask,
        num_bytes,
        cudaMemcpyDeviceToHost,
        stream.value()
    );
    stream.synchronize();
}

} // namespace cudf_shims
