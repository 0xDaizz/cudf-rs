# String Operations Module

GPU-accelerated string manipulation for string-typed `Column`s.

All operations are implemented as methods on `Column` and return new columns (or tables) without modifying the input.

## Submodules

| Module | Description | Key Methods |
|--------|-------------|-------------|
| `case` | Case conversion | `str_to_upper()`, `str_to_lower()` |
| `find` | Substring search | `str_find(target)` |
| `contains` | Containment checks | `str_contains(target)`, `str_contains_re(pattern)` |
| `replace` | String replacement | `str_replace(target, repl)`, `str_replace_re(pattern, repl)` |
| `split` | Splitting | `str_split(delimiter)` |
| `strip` | Trimming | `str_strip(chars)`, `str_lstrip(chars)`, `str_rstrip(chars)` |
| `slice` | Substring extraction | `str_slice(start, stop)` |
| `combine` | Concatenation | `str_cat(separator)` |
| `convert` | Type conversion | `str_to_integers(dtype)`, `integers_to_str()` |
| `extract` | Regex extraction | `str_extract(pattern)` |

## Examples

### Case Conversion

```rust,no_run
// Assuming `names` is a STRING column: ["alice", "bob", "charlie"]
let upper = names.str_to_upper()?;   // ["ALICE", "BOB", "CHARLIE"]
let lower = names.str_to_lower()?;   // ["alice", "bob", "charlie"]
```

### Search and Match

```rust,no_run
// Check if each string contains "error"
let has_error = log_lines.str_contains("error")?;           // BOOL8 column

// Regex match
let has_digits = log_lines.str_contains_re(r"\d+")?;        // BOOL8 column

// Find position of substring
let positions = log_lines.str_find("ERROR")?;                // INT32 column (-1 if not found)
```

### Replace

```rust,no_run
// Literal replacement
let cleaned = col.str_replace("foo", "bar")?;

// Regex replacement
let redacted = col.str_replace_re(r"\d{3}-\d{4}", "***-****")?;
```

### Split

```rust,no_run
// Split "a,b,c" by comma -> Table with 3 string columns
let parts = col.str_split(",")?;
```

### Trim

```rust,no_run
let trimmed = col.str_strip(" ")?;     // trim both sides
let ltrimmed = col.str_lstrip(" ")?;   // trim left only
let rtrimmed = col.str_rstrip(" ")?;   // trim right only
```

### Substring

```rust,no_run
// Extract characters at positions [1, 4) from each string
let sub = col.str_slice(1, 4)?;
```

### Concatenation

```rust,no_run
// Join all strings in a column with ", " separator
let joined = col.str_cat(", ")?;
```

### Type Conversion

```rust,no_run
use cudf::types::{DataType, TypeId};

// String column ["123", "456"] -> INT32 column [123, 456]
let ints = str_col.str_to_integers(DataType::new(TypeId::Int32))?;

// INT32 column [123, 456] -> String column ["123", "456"]
let strs = int_col.integers_to_str()?;
```

### Regex Extraction

```rust,no_run
// Extract first capture group from each string
// Pattern with one group: date parts from "2024-01-15"
let extracted = dates.str_extract(r"(\d{4})-\d{2}-\d{2}")?;  // ["2024", ...]
```
