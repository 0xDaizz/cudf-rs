#include "transform_shim.h"
#include <cudf/transform.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/device_buffer.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> nans_to_nulls(const OwnedColumn& col) {
    auto [new_mask, null_count] = cudf::nans_to_nulls(col.view());

    // Create a copy of the column with the new null mask applied.
    auto result = std::make_unique<cudf::column>(col.view());
    result->set_null_mask(std::move(*new_mask), null_count);
    return std::make_unique<OwnedColumn>(std::move(result));
}

rust::Vec<uint8_t> bools_to_mask(const OwnedColumn& col) {
    auto [mask_buffer, num_bytes] = cudf::bools_to_mask(col.view());

    // Copy device buffer to host.
    std::vector<uint8_t> host_data(mask_buffer->size());
    cudaMemcpy(host_data.data(), mask_buffer->data(), mask_buffer->size(), cudaMemcpyDeviceToHost);

    rust::Vec<uint8_t> out;
    out.reserve(host_data.size());
    for (auto b : host_data) {
        out.push_back(b);
    }
    return out;
}

} // namespace cudf_shims
