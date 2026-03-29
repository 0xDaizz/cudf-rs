#pragma once

#include <cudf/hashing.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Hash each row using MurmurHash3 (32-bit).
std::unique_ptr<OwnedColumn> hash_murmur3(const OwnedTable& table, uint32_t seed);

/// Hash each row using xxHash64.
std::unique_ptr<OwnedColumn> hash_xxhash64(const OwnedTable& table, uint64_t seed);

/// Hash each row using MD5 (returns a string column).
std::unique_ptr<OwnedColumn> hash_md5(const OwnedTable& table);

/// Hash each row using SHA-256 (returns a string column).
std::unique_ptr<OwnedColumn> hash_sha256(const OwnedTable& table);

} // namespace cudf_shims
