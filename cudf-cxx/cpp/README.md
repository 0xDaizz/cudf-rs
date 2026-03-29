# C++ Shim Layer

This directory contains thin C++ wrappers that adapt libcudf's C++ API for use with [cxx](https://cxx.rs).

## Why Shims?

cxx cannot directly:
- Call methods on opaque C++ types (only free functions and methods declared in the bridge)
- Handle `std::unique_ptr<T>` returns without wrapper functions
- Pass complex C++ objects (like `parquet_reader_options`) across FFI
- Handle default parameters (CUDA stream, memory resource)

Shims solve these by wrapping libcudf types in simple structs with cxx-compatible methods.

## Conventions

### Naming
- Header: `{module}_shim.h`
- Source: `{module}_shim.cpp`
- Namespace: `cudf_shims`
- Owning wrapper: `Owned{Type}` (e.g., `OwnedColumn`, `OwnedTable`)

### Exception Handling
All shim functions may throw `std::exception` (or subclasses). cxx catches these automatically when the bridge function returns `Result<T>`. Do NOT catch exceptions in shim code unless you need to transform them.

### Memory Management
- All GPU allocations use `cudf::get_current_device_resource_ref()` (RMM default)
- All operations use `cudf::get_default_stream()`
- `OwnedColumn` / `OwnedTable` destructors automatically free GPU memory via RAII
