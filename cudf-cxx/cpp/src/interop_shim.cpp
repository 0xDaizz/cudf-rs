#include "interop_shim.h"
#include <cudf/interop.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <stdexcept>

namespace cudf_shims {

namespace {

/// Convert a cudf table_view to Arrow IPC bytes via cudf's to_arrow + Arrow IPC writer.
std::vector<uint8_t> table_view_to_ipc_bytes(cudf::table_view const& tv) {
    // Build column metadata (empty names).
    cudf::column_metadata root;
    std::vector<cudf::column_metadata> col_meta(tv.num_columns());
    root.children_meta = col_meta;

    // Convert to Arrow table.
    auto arrow_table = cudf::to_arrow(tv, root.children_meta);

    // Serialize to IPC stream format in memory.
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeFileWriter(sink, arrow_table->schema()).ValueOrDie();
    auto status = writer->WriteTable(*arrow_table);
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

/// Import a cudf table from Arrow IPC bytes.
std::unique_ptr<cudf::table> table_from_ipc_bytes(const uint8_t* data, size_t size) {
    auto buf = arrow::Buffer::Wrap(data, size);
    auto buf_reader = std::make_shared<arrow::io::BufferReader>(buf);
    auto reader = arrow::ipc::RecordBatchFileReader::Open(buf_reader).ValueOrDie();

    // Read all batches into a single Arrow table.
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    for (int i = 0; i < reader->num_record_batches(); ++i) {
        batches.push_back(reader->ReadRecordBatch(i).ValueOrDie());
    }
    auto arrow_table = arrow::Table::FromRecordBatches(reader->schema(), batches).ValueOrDie();

    return cudf::from_arrow(*arrow_table);
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
