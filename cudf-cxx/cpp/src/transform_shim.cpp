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

std::unique_ptr<OwnedTable> encode_table(
    const OwnedTable& input,
    std::unique_ptr<OwnedColumn>& out_indices)
{
    auto [keys_table, indices_col] = cudf::encode(input.view());
    out_indices = std::make_unique<OwnedColumn>(std::move(indices_col));
    return std::make_unique<OwnedTable>(std::move(keys_table));
}

std::unique_ptr<OwnedTable> one_hot_encode(
    const OwnedColumn& input,
    const OwnedColumn& categories)
{
    auto [owner_col, tv] = cudf::one_hot_encode(input.view(), categories.view());

    // The table_view borrows from owner_col. We need to copy out each column
    // into an OwnedTable so it can outlive owner_col.
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(tv.num_columns());
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    for (int i = 0; i < tv.num_columns(); ++i) {
        cols.push_back(std::make_unique<cudf::column>(tv.column(i), stream, mr));
    }
    auto result = std::make_unique<cudf::table>(std::move(cols));
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> mask_to_bools(
    rust::Slice<const uint8_t> mask_data,
    int32_t begin_bit,
    int32_t end_bit)
{
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Upload mask_data to device.
    rmm::device_buffer dev_buf(mask_data.data(), mask_data.size(), stream, mr);

    auto result = cudf::mask_to_bools(
        static_cast<const cudf::bitmask_type*>(dev_buf.data()),
        begin_bit,
        end_bit);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> row_bit_count(const OwnedTable& table) {
    auto result = cudf::row_bit_count(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
