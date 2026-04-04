//! Conversion between Polars DataFrame and cudf GPU Table.
//!
//! Uses the Arrow C Data Interface as the bridge between polars-arrow and arrow-rs.

use cudf::{Column as GpuColumn, Table as GpuTable};
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_error::PolarsResult;

// Verify polars-arrow and arrow-rs FFI structs have identical memory layout.
//
// SAFETY INVARIANT: We reinterpret polars-arrow FFI structs as arrow-rs FFI structs
// (and vice versa) via `ptr::read` below. This is sound only if both struct pairs are
// `#[repr(C)]` with identical size, alignment, and field offsets. Both crates implement
// the Arrow C Data Interface spec, so the layouts should match.
//
// Field-level offset_of! verification is not possible because the upstream FFI types
// (polars-arrow and arrow-rs) have private fields (e.g. `private_data`). However,
// size + alignment equality is a sufficient check: both types implement the Arrow C
// Data Interface spec (https://arrow.apache.org/docs/format/CDataInterface.html),
// which defines a fixed C struct layout. Two `#[repr(C)]` structs with identical
// size, alignment, and the same spec-mandated field sequence are guaranteed to have
// identical field offsets.
const _: () = {
    use std::mem::{align_of, size_of};

    type PolarsArray = polars_arrow::ffi::ArrowArray;
    type ArrowArray = arrow::ffi::FFI_ArrowArray;
    type PolarsSchema = polars_arrow::ffi::ArrowSchema;
    type ArrowSchema = arrow::ffi::FFI_ArrowSchema;

    // --- ArrowArray: size and alignment ---
    assert!(size_of::<PolarsArray>() == size_of::<ArrowArray>());
    assert!(align_of::<PolarsArray>() == align_of::<ArrowArray>());

    // --- ArrowSchema: size and alignment ---
    assert!(size_of::<PolarsSchema>() == size_of::<ArrowSchema>());
    assert!(align_of::<PolarsSchema>() == align_of::<ArrowSchema>());
};

/// Convert a Polars DataFrame to a GPU-resident cudf Table.
pub fn dataframe_to_gpu(df: &DataFrame) -> PolarsResult<(GpuTable, Vec<String>)> {
    let mut gpu_columns = Vec::with_capacity(df.width());
    let mut names = Vec::with_capacity(df.width());

    for col in df.columns() {
        let series = col.as_materialized_series();
        names.push(series.name().to_string());

        // Rechunk to single contiguous array
        let series = series.rechunk();

        // Get the single arrow chunk (rechunked above, so chunk 0 has all data)
        let chunk = series.to_arrow(0, CompatLevel::oldest());

        // Export polars-arrow array via C Data Interface, import as arrow-rs
        let (ffi_array, ffi_schema) = polars_arrow_to_arrow_ffi(chunk)?;

        // Import into arrow-rs
        let arrow_data = unsafe { arrow::ffi::from_ffi(ffi_array, &ffi_schema) }
            .map_err(|e| polars_err!(ComputeError: "Arrow FFI import failed: {}", e))?;
        let arrow_array = arrow::array::make_array(arrow_data);

        // Convert arrow-rs array to GPU column
        let gpu_col = GpuColumn::from_arrow_array(arrow_array.as_ref())
            .map_err(|e| polars_err!(ComputeError: "GPU upload failed: {}", e))?;
        gpu_columns.push(gpu_col);
    }

    let table = GpuTable::new(gpu_columns)
        .map_err(|e| polars_err!(ComputeError: "GPU table creation failed: {}", e))?;
    Ok((table, names))
}

/// Convert a GPU-resident cudf Table back to a Polars DataFrame.
pub fn gpu_to_dataframe(table: GpuTable, column_names: &[String]) -> PolarsResult<DataFrame> {
    let gpu_columns = table
        .into_columns()
        .map_err(|e| polars_err!(ComputeError: "GPU column extraction failed: {}", e))?;

    if gpu_columns.len() != column_names.len() {
        return Err(polars_err!(ComputeError:
            "column count mismatch: GPU table has {} columns but {} names provided",
            gpu_columns.len(), column_names.len()));
    }
    let mut series_vec = Vec::with_capacity(gpu_columns.len());

    for (gpu_col, name) in gpu_columns.into_iter().zip(column_names) {
        // GPU column to arrow-rs array
        let arrow_array = gpu_col
            .to_arrow_array()
            .map_err(|e| polars_err!(ComputeError: "GPU download failed: {}", e))?;

        // Export arrow-rs to C Data Interface, import as polars-arrow
        let (polars_array, _polars_field) = arrow_to_polars_arrow_ffi(arrow_array.as_ref())?;

        // polars-arrow to Series
        let series = Series::from_arrow(PlSmallStr::from(name.as_str()), polars_array)?;
        series_vec.push(series.into_column());
    }

    DataFrame::new_infer_height(series_vec)
}

/// Bridge polars-arrow array to arrow-rs FFI structs.
fn polars_arrow_to_arrow_ffi(
    chunk: Box<dyn polars_arrow::array::Array>,
) -> PolarsResult<(arrow::ffi::FFI_ArrowArray, arrow::ffi::FFI_ArrowSchema)> {
    // Export polars-arrow array to C ABI structs
    let dtype = chunk.dtype().clone();
    let polars_c_array = polars_arrow::ffi::export_array_to_c(chunk);
    let field = polars_arrow::datatypes::Field::new(PlSmallStr::from_static("_"), dtype, true);
    let polars_c_schema = polars_arrow::ffi::export_field_to_c(&field);

    // Reinterpret polars-arrow C ABI structs as arrow-rs C ABI structs.
    //
    // SAFETY: Both struct pairs are `#[repr(C)]` implementations of the Arrow C Data
    // Interface. The const assertions at module level verify identical size and alignment.
    // We move ownership via ptr::read + mem::forget to avoid double-drop of the release
    // callback.
    let ffi_schema = unsafe {
        debug_assert_eq!(
            std::mem::size_of_val(&polars_c_schema),
            std::mem::size_of::<arrow::ffi::FFI_ArrowSchema>(),
            "ArrowSchema size mismatch at runtime"
        );
        // SAFETY: polars_c_schema and FFI_ArrowSchema are both #[repr(C)] Arrow C Data
        // Interface structs with identical layout (verified by const assertion + debug_assert).
        // We read-then-forget to transfer ownership without double-drop.
        std::ptr::read(
            &polars_c_schema as *const polars_arrow::ffi::ArrowSchema
                as *const arrow::ffi::FFI_ArrowSchema,
        )
    };
    // Prevent polars_c_schema from running its Drop (release callback)
    std::mem::forget(polars_c_schema);

    let ffi_array = unsafe {
        debug_assert_eq!(
            std::mem::size_of_val(&polars_c_array),
            std::mem::size_of::<arrow::ffi::FFI_ArrowArray>(),
            "ArrowArray size mismatch at runtime"
        );
        // SAFETY: polars_c_array and FFI_ArrowArray are both #[repr(C)] Arrow C Data
        // Interface structs with identical layout (verified by const assertion + debug_assert).
        // We read-then-forget to transfer ownership without double-drop.
        std::ptr::read(
            &polars_c_array as *const polars_arrow::ffi::ArrowArray
                as *const arrow::ffi::FFI_ArrowArray,
        )
    };
    std::mem::forget(polars_c_array);

    Ok((ffi_array, ffi_schema))
}

/// Bridge arrow-rs array to polars-arrow array via C ABI.
fn arrow_to_polars_arrow_ffi(
    arrow_array: &dyn arrow::array::Array,
) -> PolarsResult<(
    Box<dyn polars_arrow::array::Array>,
    polars_arrow::datatypes::Field,
)> {
    let data = arrow_array.to_data();
    let (ffi_array, ffi_schema) = arrow::ffi::to_ffi(&data)
        .map_err(|e| polars_err!(ComputeError: "Arrow FFI export failed: {}", e))?;

    // Reinterpret arrow-rs C ABI structs as polars-arrow C ABI structs.
    //
    // SAFETY: Same layout invariant as above — verified by compile-time assertions
    // at module level (size and alignment match).
    let polars_c_schema = unsafe {
        debug_assert_eq!(
            std::mem::size_of_val(&ffi_schema),
            std::mem::size_of::<polars_arrow::ffi::ArrowSchema>(),
            "ArrowSchema size mismatch at runtime"
        );
        // SAFETY: FFI_ArrowSchema and polars ArrowSchema are both #[repr(C)] Arrow C Data
        // Interface structs with identical layout (verified by const assertion + debug_assert).
        std::ptr::read(
            &ffi_schema as *const arrow::ffi::FFI_ArrowSchema
                as *const polars_arrow::ffi::ArrowSchema,
        )
    };
    std::mem::forget(ffi_schema);

    let polars_c_array = unsafe {
        debug_assert_eq!(
            std::mem::size_of_val(&ffi_array),
            std::mem::size_of::<polars_arrow::ffi::ArrowArray>(),
            "ArrowArray size mismatch at runtime"
        );
        // SAFETY: FFI_ArrowArray and polars ArrowArray are both #[repr(C)] Arrow C Data
        // Interface structs with identical layout (verified by const assertion + debug_assert).
        std::ptr::read(
            &ffi_array as *const arrow::ffi::FFI_ArrowArray as *const polars_arrow::ffi::ArrowArray,
        )
    };
    std::mem::forget(ffi_array);

    let polars_field = unsafe { polars_arrow::ffi::import_field_from_c(&polars_c_schema)? };

    let polars_array = unsafe {
        polars_arrow::ffi::import_array_from_c(polars_c_array, polars_field.dtype.clone())?
    };

    Ok((polars_array, polars_field))
}
