#include "interop_shim.h"
#include <cudf/interop.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/c/abi.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <stdexcept>

namespace cudf_shims {

namespace {

/// Convert a cudf table_view to Arrow IPC bytes via the Arrow C Data Interface.
///
/// 1. cudf::to_arrow_schema + cudf::to_arrow_host → ArrowSchema + ArrowDeviceArray (host)
/// 2. arrow::ImportRecordBatch → Arrow C++ RecordBatch
/// 3. Arrow IPC writer → serialized bytes
std::vector<uint8_t> table_view_to_ipc_bytes(cudf::table_view const& tv) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Build column metadata (empty names).
    std::vector<cudf::column_metadata> col_meta(tv.num_columns());

    // Get Arrow schema via C Data Interface.
    auto schema_ptr = cudf::to_arrow_schema(tv, col_meta);

    // Get Arrow host data via C Data Interface.
    auto device_array_ptr = cudf::to_arrow_host(tv, stream, mr);

    // Import into Arrow C++ via the bridge.
    // ArrowDeviceArray contains an ArrowArray inside it.
    auto result = arrow::ImportRecordBatch(&device_array_ptr->array, schema_ptr.get());
    if (!result.ok()) {
        throw std::runtime_error("Arrow ImportRecordBatch failed: " + result.status().ToString());
    }
    auto record_batch = result.ValueOrDie();

    // Serialize to IPC file format in memory.
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeFileWriter(sink, record_batch->schema()).ValueOrDie();
    auto status = writer->WriteRecordBatch(*record_batch);
    if (!status.ok()) {
        throw std::runtime_error("Arrow IPC write failed: " + status.ToString());
    }
    status = writer->Close();
    if (!status.ok()) {
        throw std::runtime_error("Arrow IPC close failed: " + status.ToString());
    }
    auto buffer = sink->Finish().ValueOrDie();
    const uint8_t* data = buffer->data();
    return std::vector<uint8_t>(data, data + buffer->size());
}

/// Import a cudf table from Arrow IPC bytes via the Arrow C Data Interface.
///
/// 1. Arrow IPC reader → Arrow C++ RecordBatch
/// 2. arrow::ExportRecordBatch → ArrowSchema + ArrowArray
/// 3. cudf::from_arrow(ArrowSchema*, ArrowArray*) → cudf::table
std::unique_ptr<cudf::table> table_from_ipc_bytes(const uint8_t* data, size_t size) {
    auto buf = arrow::Buffer::Wrap(data, size);
    auto buf_reader = std::make_shared<arrow::io::BufferReader>(buf);
    auto reader = arrow::ipc::RecordBatchFileReader::Open(buf_reader).ValueOrDie();

    // Read all batches into a single Arrow table, then combine into one batch.
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    for (int i = 0; i < reader->num_record_batches(); ++i) {
        batches.push_back(reader->ReadRecordBatch(i).ValueOrDie());
    }
    auto arrow_table = arrow::Table::FromRecordBatches(reader->schema(), batches).ValueOrDie();
    auto combined = arrow_table->CombineChunks().ValueOrDie();

    // Convert combined table to a single RecordBatch.
    arrow::TableBatchReader batch_reader(*combined);
    std::shared_ptr<arrow::RecordBatch> single_batch;
    auto read_status = batch_reader.ReadNext(&single_batch);
    if (!read_status.ok() || !single_batch) {
        // Empty table case.
        single_batch = arrow::RecordBatch::MakeEmpty(reader->schema()).ValueOrDie();
    }

    // Export to C Data Interface.
    ArrowSchema c_schema;
    ArrowArray c_array;
    {
        auto status = arrow::ExportRecordBatch(*single_batch, &c_array, &c_schema);
        if (!status.ok()) {
            throw std::runtime_error("Arrow ExportRecordBatch failed: " + status.ToString());
        }
    }

    return cudf::from_arrow(&c_schema, &c_array);
}

} // anonymous namespace

rust::Vec<uint8_t> column_to_arrow_ipc(const OwnedColumn& col) {
    // Wrap single column in a table_view for conversion.
    auto cv = col.view();
    auto tv = cudf::table_view({cv});
    auto bytes = table_view_to_ipc_bytes(tv);
    rust::Vec<uint8_t> out;
    out.reserve(bytes.size());
    for (auto b : bytes) {
        out.push_back(b);
    }
    return out;
}

std::unique_ptr<OwnedColumn> column_from_arrow_ipc(rust::Slice<const uint8_t> data) {
    auto table = table_from_ipc_bytes(data.data(), data.size());
    if (table->num_columns() < 1) {
        throw std::runtime_error("Arrow IPC data contains no columns");
    }
    // Extract the first column.
    auto columns = table->release();
    return std::make_unique<OwnedColumn>(std::move(columns[0]));
}

rust::Vec<uint8_t> table_to_arrow_ipc(const OwnedTable& table) {
    auto bytes = table_view_to_ipc_bytes(table.view());
    rust::Vec<uint8_t> out;
    out.reserve(bytes.size());
    for (auto b : bytes) {
        out.push_back(b);
    }
    return out;
}

std::unique_ptr<OwnedTable> table_from_arrow_ipc(rust::Slice<const uint8_t> data) {
    auto table = table_from_ipc_bytes(data.data(), data.size());
    return std::make_unique<OwnedTable>(std::move(table));
}

} // namespace cudf_shims
