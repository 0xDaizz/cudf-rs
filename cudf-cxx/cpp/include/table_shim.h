#pragma once

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/copying.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"

namespace cudf_shims {

/// Owning wrapper around std::unique_ptr<cudf::table>.
struct OwnedTable {
    std::unique_ptr<cudf::table> inner;
    // Populated lazily on first release_column call.
    std::vector<std::unique_ptr<cudf::column>> released_columns;
    bool released = false;

    explicit OwnedTable(std::unique_ptr<cudf::table> tbl)
        : inner(std::move(tbl)) {}

    int32_t num_columns() const {
        if (released) return static_cast<int32_t>(released_columns.size());
        return inner->num_columns();
    }
    int32_t num_rows() const {
        if (released) return 0;
        return inner->view().num_rows();
    }

    cudf::table_view view() const {
        if (released) throw std::runtime_error("table already released");
        return inner->view();
    }

    /// Release all columns from the table into the released_columns vector.
    /// Safe to call multiple times (idempotent).
    void ensure_released() {
        if (!released) {
            released_columns = inner->release();
            released = true;
        }
    }
};

/// Builder for constructing a table column-by-column.
/// Avoids passing Vec<UniquePtr<T>> across the cxx boundary.
struct TableBuilder {
    std::vector<std::unique_ptr<cudf::column>> columns;

    void add_column(std::unique_ptr<OwnedColumn> col) {
        if (!col) {
            throw std::runtime_error("Null column pointer in TableBuilder::add_column");
        }
        columns.push_back(std::move(col->inner));
    }

    std::unique_ptr<OwnedTable> build() {
        if (columns.empty()) {
            throw std::runtime_error("Cannot build table with zero columns");
        }
        auto table = std::make_unique<cudf::table>(std::move(columns));
        columns.clear();
        return std::make_unique<OwnedTable>(std::move(table));
    }
};

// ── Construction ───────────────────────────────────────────────

/// Create a new table builder.
std::unique_ptr<TableBuilder> table_builder_new();

// ── Column access ──────────────────────────────────────────────

/// Get a copy of a single column from the table.
std::unique_ptr<OwnedColumn> table_get_column(
    const OwnedTable& table, int32_t index);

/// Release a single column by index (extracts it from the underlying table).
std::unique_ptr<OwnedColumn> table_release_column(
    OwnedTable& table, int32_t index);

} // namespace cudf_shims
