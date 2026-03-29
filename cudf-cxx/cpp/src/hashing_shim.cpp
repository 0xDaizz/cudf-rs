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

std::unique_ptr<OwnedColumn> hash_sha1(const OwnedTable& table) {
    auto result = cudf::hashing::sha1(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> hash_sha224(const OwnedTable& table) {
    auto result = cudf::hashing::sha224(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> hash_sha384(const OwnedTable& table) {
    auto result = cudf::hashing::sha384(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> hash_sha512(const OwnedTable& table) {
    auto result = cudf::hashing::sha512(table.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> hash_xxhash32(const OwnedTable& table, uint32_t seed) {
    auto result = cudf::hashing::xxhash_32(table.view(), seed);
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> hash_murmurhash3_x64_128(const OwnedTable& table, uint64_t seed) {
    auto result = cudf::hashing::murmurhash3_x64_128(table.view(), seed);
    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
