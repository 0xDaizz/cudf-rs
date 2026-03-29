#include "interop_shim.h"
#include <cudf/interop.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/c/abi.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <arrow/c/dlpack_abi.h>

#include <stdexcept>
#include <cstring>

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

/// Helper: export a column_view to heap-allocated ArrowSchema + ArrowArray pair.
/// The schema and array are populated via cudf -> Arrow C++ -> export.
/// Caller owns both pointers and must call the release callbacks when done.
void export_column_cdata(cudf::column_view const& cv,
                         ArrowSchema** out_schema,
                         ArrowArray** out_array) {
    auto tv = cudf::table_view({cv});
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::column_metadata> col_meta(1);
    auto cudf_schema = cudf::to_arrow_schema(tv, col_meta);
    auto device_arr = cudf::to_arrow_host(tv, stream, mr);

    // Import into Arrow C++ RecordBatch.
    auto rb_result = arrow::ImportRecordBatch(&device_arr->array, cudf_schema.get());
    if (!rb_result.ok()) {
        throw std::runtime_error("Arrow ImportRecordBatch failed: " + rb_result.status().ToString());
    }
    auto rb = rb_result.ValueOrDie();

    // Extract single column array from the batch.
    if (rb->num_columns() < 1) {
        throw std::runtime_error("empty record batch after export");
    }
    auto arrow_col = rb->column(0);

    // Allocate and export via the Arrow C bridge.
    *out_schema = new ArrowSchema();
    *out_array = new ArrowArray();
    auto status = arrow::ExportArray(*arrow_col, *out_array, *out_schema);
    if (!status.ok()) {
        delete *out_schema;
        delete *out_array;
        throw std::runtime_error("Arrow ExportArray failed: " + status.ToString());
    }
}

/// Helper: export a table_view to heap-allocated ArrowSchema + ArrowArray.
void export_table_cdata(cudf::table_view const& tv,
                        ArrowSchema** out_schema,
                        ArrowArray** out_array) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::column_metadata> col_meta(tv.num_columns());
    auto cudf_schema = cudf::to_arrow_schema(tv, col_meta);
    auto device_arr = cudf::to_arrow_host(tv, stream, mr);

    auto rb_result = arrow::ImportRecordBatch(&device_arr->array, cudf_schema.get());
    if (!rb_result.ok()) {
        throw std::runtime_error("Arrow ImportRecordBatch failed: " + rb_result.status().ToString());
    }
    auto rb = rb_result.ValueOrDie();

    *out_schema = new ArrowSchema();
    *out_array = new ArrowArray();
    auto status = arrow::ExportRecordBatch(*rb, *out_array, *out_schema);
    if (!status.ok()) {
        delete *out_schema;
        delete *out_array;
        throw std::runtime_error("Arrow ExportRecordBatch failed: " + status.ToString());
    }
}

/// State holder for contiguous_split results.
struct SplitResult {
    std::vector<cudf::packed_table> parts;
};

} // anonymous namespace

// ── Arrow IPC ──────────────────────────────────────────────────

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

// ── Arrow C Data Interface ────────────────────────────────────

uint64_t column_to_arrow_schema_ptr(const OwnedColumn& col) {
    ArrowSchema* schema = nullptr;
    ArrowArray* array = nullptr;
    export_column_cdata(col.view(), &schema, &array);
    // We export both but only return schema here; the array is leaked
    // intentionally — the caller must also call column_to_arrow_array_ptr.
    // To avoid double-export we cache the pair.
    // Actually, cleaner: export once, return both via two calls with a shared export.
    // But cxx doesn't support tuples. So we export twice (cheap for host data).
    // Free the array from this call since it won't be used.
    if (array->release) {
        array->release(array);
    }
    delete array;
    return reinterpret_cast<uint64_t>(schema);
}

uint64_t column_to_arrow_array_ptr(const OwnedColumn& col) {
    ArrowSchema* schema = nullptr;
    ArrowArray* array = nullptr;
    export_column_cdata(col.view(), &schema, &array);
    // Free the schema from this call.
    if (schema->release) {
        schema->release(schema);
    }
    delete schema;
    return reinterpret_cast<uint64_t>(array);
}

std::unique_ptr<OwnedColumn> column_from_arrow_cdata(uint64_t schema_ptr, uint64_t array_ptr) {
    if (schema_ptr == 0 || array_ptr == 0) {
        throw std::runtime_error("null Arrow schema or array pointer");
    }
    auto* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    auto* array = reinterpret_cast<ArrowArray*>(array_ptr);

    // Wrap the single column in a struct schema for cudf::from_arrow (table-level).
    // cudf::from_arrow expects a struct (table) schema with child columns.
    ArrowSchema struct_schema;
    std::memset(&struct_schema, 0, sizeof(struct_schema));
    struct_schema.format = "+s";
    struct_schema.name = "";
    struct_schema.n_children = 1;
    auto* children_ptrs = new ArrowSchema*[1];
    children_ptrs[0] = schema;
    struct_schema.children = children_ptrs;
    struct_schema.release = [](ArrowSchema* s) {
        // Release children pointer array, but NOT the child schemas themselves
        // (they are owned by the original schema pointer).
        delete[] s->children;
        s->children = nullptr;
        s->release = nullptr;
    };

    ArrowArray struct_array;
    std::memset(&struct_array, 0, sizeof(struct_array));
    struct_array.length = array->length;
    struct_array.null_count = 0;
    struct_array.offset = 0;
    struct_array.n_buffers = 1;
    // Struct array needs a single null buffer (nullptr = all valid).
    auto* null_buf = new const void*[1];
    null_buf[0] = nullptr;
    struct_array.buffers = null_buf;
    struct_array.n_children = 1;
    auto* arr_children = new ArrowArray*[1];
    arr_children[0] = array;
    struct_array.children = arr_children;
    struct_array.release = [](ArrowArray* a) {
        delete[] a->buffers;
        delete[] a->children;
        a->buffers = nullptr;
        a->children = nullptr;
        a->release = nullptr;
    };

    auto table = cudf::from_arrow(&struct_schema, &struct_array);

    // Release the original schema and array.
    if (schema->release) {
        schema->release(schema);
    }
    delete schema;
    if (array->release) {
        array->release(array);
    }
    delete array;

    if (table->num_columns() < 1) {
        throw std::runtime_error("from_arrow produced no columns");
    }
    auto columns = table->release();
    return std::make_unique<OwnedColumn>(std::move(columns[0]));
}

uint64_t table_to_arrow_schema_ptr(const OwnedTable& table) {
    ArrowSchema* schema = nullptr;
    ArrowArray* array = nullptr;
    export_table_cdata(table.view(), &schema, &array);
    if (array->release) {
        array->release(array);
    }
    delete array;
    return reinterpret_cast<uint64_t>(schema);
}

uint64_t table_to_arrow_array_ptr(const OwnedTable& table) {
    ArrowSchema* schema = nullptr;
    ArrowArray* array = nullptr;
    export_table_cdata(table.view(), &schema, &array);
    if (schema->release) {
        schema->release(schema);
    }
    delete schema;
    return reinterpret_cast<uint64_t>(array);
}

std::unique_ptr<OwnedTable> table_from_arrow_cdata(uint64_t schema_ptr, uint64_t array_ptr) {
    if (schema_ptr == 0 || array_ptr == 0) {
        throw std::runtime_error("null Arrow schema or array pointer");
    }
    auto* schema = reinterpret_cast<ArrowSchema*>(schema_ptr);
    auto* array = reinterpret_cast<ArrowArray*>(array_ptr);

    auto table = cudf::from_arrow(schema, array);

    if (schema->release) {
        schema->release(schema);
    }
    delete schema;
    if (array->release) {
        array->release(array);
    }
    delete array;

    return std::make_unique<OwnedTable>(std::move(table));
}

void free_arrow_schema(uint64_t ptr) {
    if (ptr == 0) return;
    auto* schema = reinterpret_cast<ArrowSchema*>(ptr);
    if (schema->release) {
        schema->release(schema);
    }
    delete schema;
}

void free_arrow_array(uint64_t ptr) {
    if (ptr == 0) return;
    auto* array = reinterpret_cast<ArrowArray*>(ptr);
    if (array->release) {
        array->release(array);
    }
    delete array;
}

// ── Arrow C Data Interface (paired export) ───────────────────

std::unique_ptr<ArrowExportPair> column_to_arrow_pair(const OwnedColumn& col) {
    auto pair = std::make_unique<ArrowExportPair>();
    export_column_cdata(col.view(), &pair->schema, &pair->array);
    return pair;
}

std::unique_ptr<ArrowExportPair> table_to_arrow_pair(const OwnedTable& table) {
    auto pair = std::make_unique<ArrowExportPair>();
    export_table_cdata(table.view(), &pair->schema, &pair->array);
    return pair;
}

uint64_t arrow_pair_schema(ArrowExportPair& pair) {
    auto* ptr = pair.schema;
    pair.schema = nullptr;  // release ownership so destructor won't double-free
    return reinterpret_cast<uint64_t>(ptr);
}

uint64_t arrow_pair_array(ArrowExportPair& pair) {
    auto* ptr = pair.array;
    pair.array = nullptr;  // release ownership so destructor won't double-free
    return reinterpret_cast<uint64_t>(ptr);
}

// ── DLPack ────────────────────────────────────────────────────

uint64_t table_to_dlpack(const OwnedTable& table) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    DLManagedTensor* tensor = cudf::to_dlpack(table.view(), stream, mr);
    return reinterpret_cast<uint64_t>(tensor);
}

std::unique_ptr<OwnedTable> table_from_dlpack(uint64_t dlpack_ptr) {
    if (dlpack_ptr == 0) {
        throw std::runtime_error("null DLPack tensor pointer");
    }
    auto* tensor = reinterpret_cast<DLManagedTensor*>(dlpack_ptr);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto table = cudf::from_dlpack(tensor, stream, mr);
    return std::make_unique<OwnedTable>(std::move(table));
}

void free_dlpack(uint64_t dlpack_ptr) {
    if (dlpack_ptr == 0) return;
    auto* tensor = reinterpret_cast<DLManagedTensor*>(dlpack_ptr);
    if (tensor->deleter) {
        tensor->deleter(tensor);
    }
}

// ── contiguous_split / pack / unpack ──────────────────────────

std::unique_ptr<OwnedPackedColumns> pack_table(const OwnedTable& table) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto packed = cudf::pack(table.view(), stream, mr);
    return std::make_unique<OwnedPackedColumns>(std::move(packed));
}

rust::Vec<uint8_t> packed_metadata(const OwnedPackedColumns& packed) {
    rust::Vec<uint8_t> out;
    if (packed.inner.metadata) {
        auto& md = *packed.inner.metadata;
        out.reserve(md.size());
        for (auto b : md) {
            out.push_back(b);
        }
    }
    return out;
}

int64_t packed_gpu_data_size(const OwnedPackedColumns& packed) {
    if (packed.inner.gpu_data) {
        return static_cast<int64_t>(packed.inner.gpu_data->size());
    }
    return 0;
}

std::unique_ptr<OwnedTable> unpack_table(const OwnedPackedColumns& packed) {
    // unpack returns a table_view that borrows from packed.inner.
    // We must deep-copy it into an owned table.
    auto tv = cudf::unpack(packed.inner);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Deep copy each column.
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(tv.num_columns());
    for (cudf::size_type i = 0; i < tv.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(tv.column(i), stream, mr));
    }
    auto table = std::make_unique<cudf::table>(std::move(columns));
    return std::make_unique<OwnedTable>(std::move(table));
}

rust::Vec<uint64_t> contiguous_split_table(
    const OwnedTable& table,
    rust::Slice<const int32_t> splits) {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    std::vector<cudf::size_type> split_vec(splits.begin(), splits.end());
    auto parts = cudf::contiguous_split(table.view(), split_vec, stream, mr);

    // Store the result on the heap so Rust can access individual parts.
    auto* result = new SplitResult{std::move(parts)};

    // Return a vector with: [handle, num_parts].
    // The handle is used with contiguous_split_get / contiguous_split_free.
    rust::Vec<uint64_t> out;
    out.push_back(reinterpret_cast<uint64_t>(result));
    out.push_back(static_cast<uint64_t>(result->parts.size()));
    return out;
}

std::unique_ptr<OwnedPackedColumns> contiguous_split_get(uint64_t handle, int32_t index) {
    if (handle == 0) {
        throw std::runtime_error("null contiguous_split handle");
    }
    auto* result = reinterpret_cast<SplitResult*>(handle);
    if (index < 0 || static_cast<size_t>(index) >= result->parts.size()) {
        throw std::runtime_error("contiguous_split index out of range");
    }

    // Move the packed data out of the split result.
    auto packed = std::move(result->parts[index].data);
    return std::make_unique<OwnedPackedColumns>(std::move(packed));
}

void contiguous_split_free(uint64_t handle) {
    if (handle == 0) return;
    auto* result = reinterpret_cast<SplitResult*>(handle);
    delete result;
}

} // namespace cudf_shims
