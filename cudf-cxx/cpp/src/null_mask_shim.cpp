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

} // namespace cudf_shims
