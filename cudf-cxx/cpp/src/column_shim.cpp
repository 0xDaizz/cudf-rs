#include "column_shim.h"
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <limits>

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

std::unique_ptr<OwnedColumn> column_from_strings(rust::Slice<const rust::String> data) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto num_strings = static_cast<cudf::size_type>(data.size());

    if (num_strings == 0) {
        // Build an empty strings column via offsets + empty chars.
        auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32},
            1,
            rmm::device_buffer(sizeof(int32_t), stream, mr),
            rmm::device_buffer{}, 0);
        // Zero out the single offset.
        int32_t zero = 0;
        auto err = cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(), &zero,
                        sizeof(int32_t), cudaMemcpyHostToDevice, stream.value());
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(err));
        }
        stream.synchronize();
        return std::make_unique<OwnedColumn>(
            cudf::make_strings_column(
                0, std::move(offsets_col),
                rmm::device_buffer{0, stream, mr}, 0,
                rmm::device_buffer{0, stream, mr}));
    }

    // Guard: total string data must fit in int32_t offsets (2GB limit).
    size_t total_chars = 0;
    for (const auto& s : data) {
        total_chars += s.size();
    }
    if (total_chars > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Total string data exceeds INT32_MAX bytes (2GB limit)");
    }

    // Build concatenated char buffer and offsets on host.
    std::vector<int32_t> offsets_vec;
    offsets_vec.reserve(num_strings + 1);
    std::string combined;
    int32_t offset = 0;
    offsets_vec.push_back(0);
    for (const auto& s : data) {
        combined.append(s.data(), s.size());
        offset += static_cast<int32_t>(s.size());
        offsets_vec.push_back(offset);
    }

    // Upload chars to device.
    rmm::device_buffer chars_buf(combined.data(), combined.size(), stream, mr);

    // Upload offsets to device as a cudf::column.
    auto offsets_byte_size = offsets_vec.size() * sizeof(int32_t);
    rmm::device_buffer offsets_buf(offsets_vec.data(), offsets_byte_size, stream, mr);
    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(offsets_vec.size()),
        std::move(offsets_buf),
        rmm::device_buffer{}, 0);

    auto col = cudf::make_strings_column(
        num_strings, std::move(offsets_col),
        std::move(chars_buf), 0, rmm::device_buffer{0, stream, mr});

    return std::make_unique<OwnedColumn>(std::move(col));
}

// ── Nullable column helpers ───────────────────────────────────

namespace {

/// Build a validity bitmask from a bool slice and upload to device.
/// Returns {device_buffer, null_count}.
std::pair<rmm::device_buffer, cudf::size_type> build_null_mask(
    rust::Slice<const bool> validity, cudf::size_type size,
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
    auto num_bytes = cudf::bitmask_allocation_size_bytes(size);
    std::vector<uint8_t> host_mask(num_bytes, 0);
    cudf::size_type null_count = 0;
    for (cudf::size_type i = 0; i < size; ++i) {
        if (validity[i]) {
            host_mask[i / 8] |= (1u << (i % 8));
        } else {
            ++null_count;
        }
    }
    rmm::device_buffer dev_mask(host_mask.data(), num_bytes, stream, mr);
    return {std::move(dev_mask), null_count};
}

template <typename T>
std::unique_ptr<OwnedColumn> column_from_nullable(
    rust::Slice<const T> data,
    rust::Slice<const bool> validity,
    cudf::type_id tid)
{
    if (data.size() != validity.size()) {
        throw std::runtime_error("data and validity slices must have the same length");
    }
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto size = static_cast<cudf::size_type>(data.size());
    auto byte_size = size * sizeof(T);

    rmm::device_buffer dev_buf(data.data(), byte_size, stream, mr);
    auto [mask, null_count] = build_null_mask(validity, size, stream, mr);

    auto col = std::make_unique<cudf::column>(
        cudf::data_type{tid},
        size,
        std::move(dev_buf),
        std::move(mask),
        null_count);

    return std::make_unique<OwnedColumn>(std::move(col));
}

} // anonymous namespace

std::unique_ptr<OwnedColumn> column_from_i32_nullable(
    rust::Slice<const int32_t> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::INT32);
}

std::unique_ptr<OwnedColumn> column_from_i64_nullable(
    rust::Slice<const int64_t> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::INT64);
}

std::unique_ptr<OwnedColumn> column_from_f32_nullable(
    rust::Slice<const float> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::FLOAT32);
}

std::unique_ptr<OwnedColumn> column_from_f64_nullable(
    rust::Slice<const double> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::FLOAT64);
}

std::unique_ptr<OwnedColumn> column_from_i8_nullable(
    rust::Slice<const int8_t> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::INT8);
}

std::unique_ptr<OwnedColumn> column_from_i16_nullable(
    rust::Slice<const int16_t> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::INT16);
}

std::unique_ptr<OwnedColumn> column_from_u8_nullable(
    rust::Slice<const uint8_t> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::UINT8);
}

std::unique_ptr<OwnedColumn> column_from_u16_nullable(
    rust::Slice<const uint16_t> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::UINT16);
}

std::unique_ptr<OwnedColumn> column_from_u32_nullable(
    rust::Slice<const uint32_t> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::UINT32);
}

std::unique_ptr<OwnedColumn> column_from_u64_nullable(
    rust::Slice<const uint64_t> data, rust::Slice<const bool> validity) {
    return column_from_nullable(data, validity, cudf::type_id::UINT64);
}

std::unique_ptr<OwnedColumn> column_from_bool_nullable(
    rust::Slice<const bool> data, rust::Slice<const bool> validity) {
    if (data.size() != validity.size()) {
        throw std::runtime_error("data and validity slices must have the same length");
    }
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto size = static_cast<cudf::size_type>(data.size());

    // Convert bool -> int8_t (cudf stores BOOL8 as int8)
    std::vector<int8_t> int_data(size);
    for (cudf::size_type i = 0; i < size; ++i) {
        int_data[i] = data[i] ? 1 : 0;
    }

    rmm::device_buffer dev_buf(int_data.data(), size * sizeof(int8_t), stream, mr);
    auto [mask, null_count] = build_null_mask(validity, size, stream, mr);

    auto col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::BOOL8},
        size,
        std::move(dev_buf),
        std::move(mask),
        null_count);

    return std::make_unique<OwnedColumn>(std::move(col));
}

std::unique_ptr<OwnedColumn> column_from_strings_nullable(
    rust::Slice<const rust::String> data,
    rust::Slice<const bool> validity)
{
    if (data.size() != validity.size()) {
        throw std::runtime_error("data and validity slices must have the same length");
    }
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto num_strings = static_cast<cudf::size_type>(data.size());

    if (num_strings == 0) {
        // Build an empty strings column.
        auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32},
            1,
            rmm::device_buffer(sizeof(int32_t), stream, mr),
            rmm::device_buffer{}, 0);
        int32_t zero = 0;
        auto err = cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(), &zero,
                        sizeof(int32_t), cudaMemcpyHostToDevice, stream.value());
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(err));
        }
        stream.synchronize();
        return std::make_unique<OwnedColumn>(
            cudf::make_strings_column(
                0, std::move(offsets_col),
                rmm::device_buffer{0, stream, mr}, 0,
                rmm::device_buffer{0, stream, mr}));
    }

    // Guard: total string data must fit in int32_t offsets (2GB limit).
    size_t total_chars = 0;
    for (const auto& s : data) {
        total_chars += s.size();
    }
    if (total_chars > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Total string data exceeds INT32_MAX bytes (2GB limit)");
    }

    // Build concatenated char buffer and offsets on host.
    std::vector<int32_t> offsets_vec;
    offsets_vec.reserve(num_strings + 1);
    std::string combined;
    int32_t offset = 0;
    offsets_vec.push_back(0);
    for (const auto& s : data) {
        combined.append(s.data(), s.size());
        offset += static_cast<int32_t>(s.size());
        offsets_vec.push_back(offset);
    }

    // Upload chars to device.
    rmm::device_buffer chars_buf(combined.data(), combined.size(), stream, mr);

    // Upload offsets to device as a cudf::column.
    auto offsets_byte_size = offsets_vec.size() * sizeof(int32_t);
    rmm::device_buffer offsets_buf(offsets_vec.data(), offsets_byte_size, stream, mr);
    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(offsets_vec.size()),
        std::move(offsets_buf),
        rmm::device_buffer{}, 0);

    // Build null mask.
    auto [mask, null_count] = build_null_mask(validity, num_strings, stream, mr);

    auto col = cudf::make_strings_column(
        num_strings, std::move(offsets_col),
        std::move(chars_buf), null_count, std::move(mask));

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
    auto err = cudaMemcpyAsync(
        out.data(),
        view.data<T>(),
        view.size() * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream.value()
    );
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(err));
    }
    stream.synchronize();
}

void column_to_i8(const OwnedColumn& col, rust::Slice<int8_t> out) {
    column_to_host(col, out);
}

void column_to_i16(const OwnedColumn& col, rust::Slice<int16_t> out) {
    column_to_host(col, out);
}

void column_to_i32(const OwnedColumn& col, rust::Slice<int32_t> out) {
    column_to_host(col, out);
}

void column_to_i64(const OwnedColumn& col, rust::Slice<int64_t> out) {
    column_to_host(col, out);
}

void column_to_u8(const OwnedColumn& col, rust::Slice<uint8_t> out) {
    column_to_host(col, out);
}

void column_to_u16(const OwnedColumn& col, rust::Slice<uint16_t> out) {
    column_to_host(col, out);
}

void column_to_u32(const OwnedColumn& col, rust::Slice<uint32_t> out) {
    column_to_host(col, out);
}

void column_to_u64(const OwnedColumn& col, rust::Slice<uint64_t> out) {
    column_to_host(col, out);
}

void column_to_f32(const OwnedColumn& col, rust::Slice<float> out) {
    column_to_host(col, out);
}

void column_to_f64(const OwnedColumn& col, rust::Slice<double> out) {
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
    auto err = cudaMemcpyAsync(
        out.data(),
        mask,
        num_bytes,
        cudaMemcpyDeviceToHost,
        stream.value()
    );
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(err));
    }
    stream.synchronize();
}

rust::Vec<rust::String> column_to_strings(const OwnedColumn& col) {
    auto view = col.view();
    if (view.type().id() != cudf::type_id::STRING) {
        throw std::runtime_error("column_to_strings: column is not a STRING type");
    }

    auto stream = cudf::get_default_stream();
    auto num_strings = view.size();
    rust::Vec<rust::String> result;

    if (num_strings == 0) {
        return result;
    }

    auto strings_view = cudf::strings_column_view(view);

    // Get offsets from device.
    auto offsets_view = strings_view.offsets();
    auto offsets_size = (num_strings + 1) * sizeof(int32_t);
    std::vector<int32_t> host_offsets(num_strings + 1);
    auto offset_data = offsets_view.data<int32_t>() + strings_view.offset();
    auto err = cudaMemcpyAsync(
        host_offsets.data(), offset_data,
        offsets_size, cudaMemcpyDeviceToHost, stream.value());
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync offsets failed: ") + cudaGetErrorString(err));
    }

    // Get chars from device.
    // Synchronize to ensure offsets are available before computing chars_size.
    stream.synchronize();
    auto chars_size = host_offsets.back() - host_offsets.front();

    std::vector<char> host_chars(chars_size);
    if (chars_size > 0) {
        auto chars_data = strings_view.chars_begin(stream);
        err = cudaMemcpyAsync(
            host_chars.data(), chars_data,
            chars_size, cudaMemcpyDeviceToHost, stream.value());
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync chars failed: ") + cudaGetErrorString(err));
        }
    }

    // Get null mask if nullable.
    std::vector<uint8_t> host_mask;
    bool has_nulls = view.nullable() && view.null_count() > 0;
    if (has_nulls) {
        auto mask_bytes = cudf::bitmask_allocation_size_bytes(num_strings);
        host_mask.resize(mask_bytes);
        err = cudaMemcpyAsync(
            host_mask.data(), view.null_mask(),
            mask_bytes, cudaMemcpyDeviceToHost, stream.value());
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync mask failed: ") + cudaGetErrorString(err));
        }
    }

    stream.synchronize();

    auto base_offset = host_offsets.front();
    for (cudf::size_type i = 0; i < num_strings; ++i) {
        if (has_nulls && !(host_mask[i / 8] & (1u << (i % 8)))) {
            // Null entry -- push empty string (caller checks null mask separately)
            result.push_back(rust::String());
        } else {
            auto start = host_offsets[i] - base_offset;
            auto len = host_offsets[i + 1] - host_offsets[i];
            result.push_back(rust::String(host_chars.data() + start, len));
        }
    }

    return result;
}

} // namespace cudf_shims
