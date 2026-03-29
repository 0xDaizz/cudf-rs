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

/// Hash each row using SHA-1 (returns a string column).
std::unique_ptr<OwnedColumn> hash_sha1(const OwnedTable& table);

/// Hash each row using SHA-224 (returns a string column).
std::unique_ptr<OwnedColumn> hash_sha224(const OwnedTable& table);

/// Hash each row using SHA-384 (returns a string column).
std::unique_ptr<OwnedColumn> hash_sha384(const OwnedTable& table);

/// Hash each row using SHA-512 (returns a string column).
std::unique_ptr<OwnedColumn> hash_sha512(const OwnedTable& table);

/// Hash each row using xxHash32.
std::unique_ptr<OwnedColumn> hash_xxhash32(const OwnedTable& table, uint32_t seed);

/// Hash each row using MurmurHash3 x64 128-bit. Returns a 2-column table.
std::unique_ptr<OwnedTable> hash_murmurhash3_x64_128(const OwnedTable& table, uint64_t seed);

} // namespace cudf_shims
