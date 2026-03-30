#include "null_mask_shim.h"
#include <cudf/null_mask.hpp>
#include <cudf/copying.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace cudf_shims {

// ── Existing functions (Phase 1) ─────────────────────────────

int32_t valid_count(const OwnedColumn& col) {
    auto view = col.view();
    if (!view.nullable()) return view.size();
    return view.size() - view.null_count();
}

std::unique_ptr<OwnedColumn> set_all_valid(const OwnedColumn& col) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    // Copy the column then strip its null mask
    auto copied = std::make_unique<cudf::column>(col.view(), stream, mr);
    copied->set_null_mask(rmm::device_buffer{}, 0);
    return std::make_unique<OwnedColumn>(std::move(copied));
}

// ── New functions (Phase 2) ──────────────────────────────────

namespace {

cudf::mask_state to_mask_state(int32_t state) {
    switch (state) {
        case 0: return cudf::mask_state::UNALLOCATED;
        case 1: return cudf::mask_state::UNINITIALIZED;
        case 2: return cudf::mask_state::ALL_VALID;
        case 3: return cudf::mask_state::ALL_NULL;
        default: throw std::runtime_error("Invalid mask_state: " + std::to_string(state));
    }
}

} // anonymous namespace

std::unique_ptr<OwnedDeviceBuffer> create_null_mask(
    int32_t size, int32_t state)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto buf = cudf::create_null_mask(size, to_mask_state(state), stream, mr);
    return std::make_unique<OwnedDeviceBuffer>(std::move(buf));
}

int32_t null_count_column(const OwnedColumn& col) {
    return col.null_count();
}

int32_t bitmask_allocation_size(int32_t number_of_bits) {
    return static_cast<int32_t>(cudf::bitmask_allocation_size_bytes(number_of_bits));
}

void copy_null_mask_to_host(
    const OwnedColumn& col,
    rust::Slice<uint8_t> out)
{
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

std::unique_ptr<OwnedColumn> set_null_mask_from_host(
    const OwnedColumn& col,
    rust::Slice<const uint8_t> mask,
    int32_t null_count)
{
    auto view = col.view();
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Copy the column first
    auto new_col = std::make_unique<cudf::column>(view, stream, mr);

    // Copy mask to device
    auto mask_size = cudf::bitmask_allocation_size_bytes(view.size());
    rmm::device_buffer dev_mask(mask.data(), mask_size, stream, mr);

    // Set the null mask on the new column
    new_col->set_null_mask(std::move(dev_mask), null_count);

    return std::make_unique<OwnedColumn>(std::move(new_col));
}

std::unique_ptr<OwnedColumn> set_null_mask_range(
    const OwnedColumn& col,
    int32_t begin_bit,
    int32_t end_bit,
    bool valid)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Copy the column.
    auto new_col = std::make_unique<cudf::column>(col.view(), stream, mr);

    // If not nullable, allocate a mask first.
    if (!new_col->nullable()) {
        auto mask_buf = cudf::create_null_mask(new_col->size(), cudf::mask_state::ALL_VALID, stream, mr);
        new_col->set_null_mask(std::move(mask_buf), 0);
    }

    // Set the range.
    cudf::set_null_mask(
        static_cast<cudf::bitmask_type*>(new_col->mutable_view().null_mask()),
        begin_bit, end_bit, valid, stream);
    stream.synchronize();

    // Recount nulls: count unset bits in the mask range [0, size).
    auto view = new_col->view();
    auto null_cnt = cudf::null_count(
        view.null_mask(), 0, view.size());
    new_col->set_null_count(static_cast<cudf::size_type>(null_cnt));

    return std::make_unique<OwnedColumn>(std::move(new_col));
}

rust::Vec<uint8_t> copy_bitmask_to_host(const OwnedColumn& col) {
    rust::Vec<uint8_t> out;
    auto view = col.view();
    if (!view.nullable()) {
        return out;
    }

    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto dev_buf = cudf::copy_bitmask(view, stream, mr);

    auto num_bytes = dev_buf.size();
    std::vector<uint8_t> host_data(num_bytes);
    cudaMemcpyAsync(host_data.data(), dev_buf.data(), num_bytes, cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();

    out.reserve(num_bytes);
    for (auto b : host_data) {
        out.push_back(b);
    }
    return out;
}

std::unique_ptr<BitmaskBuilder> bitmask_builder_new() {
    return std::make_unique<BitmaskBuilder>();
}

std::unique_ptr<BitmaskResult> bitmask_and(const BitmaskBuilder& builder) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    cudf::table_view tv(builder.views);
    auto [dev_buf, null_count] = cudf::bitmask_and(tv, stream, mr);

    std::vector<uint8_t> host_data(dev_buf.size());
    cudaMemcpyAsync(host_data.data(), dev_buf.data(), dev_buf.size(), cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();

    auto result = std::make_unique<BitmaskResult>();
    result->mask.reserve(host_data.size());
    for (auto b : host_data) {
        result->mask.push_back(b);
    }
    result->null_count = null_count;
    return result;
}

std::unique_ptr<BitmaskResult> bitmask_or(const BitmaskBuilder& builder) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    cudf::table_view tv(builder.views);
    auto [dev_buf, null_count] = cudf::bitmask_or(tv, stream, mr);

    std::vector<uint8_t> host_data(dev_buf.size());
    cudaMemcpyAsync(host_data.data(), dev_buf.data(), dev_buf.size(), cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();

    auto result = std::make_unique<BitmaskResult>();
    result->mask.reserve(host_data.size());
    for (auto b : host_data) {
        result->mask.push_back(b);
    }
    result->null_count = null_count;
    return result;
}

int32_t state_null_count(int32_t state, int32_t size)
{
    auto ms = to_mask_state(state);
    return cudf::state_null_count(ms, size);
}

int32_t num_bitmask_words(int32_t number_of_bits)
{
    return cudf::num_bitmask_words(number_of_bits);
}

} // namespace cudf_shims
