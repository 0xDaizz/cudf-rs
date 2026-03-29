#include "hashing_shim.h"
#include <cudf/hashing.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace cudf_shims {

std::unique_ptr<OwnedColumn> hash_murmur3(const OwnedTable& table, uint32_t seed) {
    auto result = cudf::hashing::murmurhash3_x86_32(table.view(), seed);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> hash_xxhash64(const OwnedTable& table, uint64_t seed) {
    auto result = cudf::hashing::xxhash_64(table.view(), seed);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> hash_md5(const OwnedTable& table) {
    auto result = cudf::hashing::md5(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> hash_sha256(const OwnedTable& table) {
    auto result = cudf::hashing::sha256(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
