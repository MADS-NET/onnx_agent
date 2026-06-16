# Chronos-2 ONNX Example

Time-series forecasting of the `chicken.csv` dataset using the
[kashif/chronos-2-onnx](https://huggingface.co/kashif/chronos-2-onnx) model
and the `mads-onnx-filter` agent.

---

## Discovered tensor schema

Inspected from `model_fixed.onnx` via `mads-onnx-filter --inspect`.

### Inputs

| Name                | dtype   | shape                   | Notes                                              |
|---------------------|---------|-------------------------|----------------------------------------------------|
| `context`           | float32 | `[batch, context_len]`  | Historical time series values (univariate)         |
| `group_ids`         | int64   | `[batch]`               | Series group identifier; use 0 for a single series |
| `attention_mask`    | float32 | `[batch, context_len]`  | 1 = real observation, 0 = left-pad position        |
| `future_covariates` | float32 | `[batch, future_len]`   | Future covariate features; zeros when not available|
| `num_output_patches`| int64   | `[]` (scalar)           | Number of output patches (patch size = 16)         |

### Output

| Name             | dtype   | shape                      | Notes                                             |
|------------------|---------|----------------------------|---------------------------------------------------|
| `quantile_preds` | float32 | `[batch, 21, future_len]`  | 21 quantile levels: 0.01, 0.06, … 0.94, 0.99    |

**Quantile index 10 (q = 0.50) is the median forecast.**

`future_len = num_output_patches * patch_size = num_output_patches * 16`

---

## Model quirks and preprocessing notes

1. **context_length must be a multiple of the patch size (16).**  
   The model internally reshapes `context` to `[batch, -1, 16]`.  
   Left-pad shorter series with zeros and set `attention_mask = 0` for those
   positions.  `make_input.py` handles this automatically.

2. **The released ONNX files contain a graph bug.**  
   A `ConstantOfShape` node emits `float32` values that are used as integer
   indices by a downstream `Gather` node.  ONNX Runtime rejects this.  
   `fix_model.py` inserts a `Cast(to=INT64)` node to fix it in-place and
   saves the result as `model_fixed.onnx`.

3. **No built-in scaling / normalisation.**  
   The model accepts raw float values; the arcsinh transformation described in
   the config is applied internally by the graph (not as a separate step).

4. **Single-series, batch size 1 is the typical usage.**  
   The model supports arbitrary batch sizes but the example uses batch = 1.

---

## Step-by-step usage

All commands are run from the **project root** (`/path/to/onnx_agent`).

### 1. Build the agent (if not already done)

```bash
cmake -Bbuild && cmake --build build -j6
```

### 2. Download and fix the model

```bash
pip3 install onnx          # required by fix_model.py
bash examples/chronos/download_model.sh
```

This downloads `model.onnx` (~456 MB) from HuggingFace and produces
`model_fixed.onnx` with the graph bug patched.

**Windows equivalent (PowerShell):**

```powershell
Invoke-WebRequest `
  -Uri "https://huggingface.co/kashif/chronos-2-onnx/resolve/main/model.onnx" `
  -OutFile "examples\chronos\model.onnx"
pip install onnx
python examples\chronos\fix_model.py
```

### 3. Inspect the model

```bash
./build/src/main/mads-onnx-filter \
  --inspect -m examples/chronos/model_fixed.onnx
```

### 4. Generate the input JSON from chicken.csv

```bash
python3 examples/chronos/make_input.py
# writes examples/chronos/chronos_input.json
```

Options:

```
--csv PATH               path to chicken.csv (auto-detected by default)
--out PATH               output JSON path (default: examples/chronos/chronos_input.json)
--prediction-months N    forecast horizon in months (default: 12, rounded up to 16)
```

### 5. Run the agent in offline test mode

```bash
./build/src/main/mads-onnx-filter \
  --test \
  -m examples/chronos/model_fixed.onnx \
  --input examples/chronos/chronos_input.json
```

Expected output (abridged):

```
=== mads-onnx-filter --test ===
Model: examples/chronos/model_fixed.onnx

Input file: examples/chronos/chronos_input.json

Output:
{
  "quantile_preds": {
    "dtype": "float32",
    "shape": [1, 21, 16],
    "data": [17.91, 14.56, ..., 167.70]   // 21 * 16 = 336 values
  }
}
```

Shape `[1, 21, 16]` means: 1 batch × 21 quantiles × 16 future months.  
The median forecast (index 10) for the next 16 months starting after
the last chicken data point is approximately 65–73 (in the same units as
the `value` column).

---

## Live agent usage (MADS broker mode)

When running against a live MADS broker, the agent subscribes to the
`onnx-filter-input` topic and publishes results to `onnx-filter`:

```bash
./build/src/main/mads-onnx-filter \
  -m examples/chronos/model_fixed.onnx
```

Publishers on `onnx-filter-input` must send JSON messages with the same
tensor schema used in `chronos_input.json`:

```json
{
  "context":            {"shape": [1, 192], "data": [...], "dtype": "float32"},
  "group_ids":          {"shape": [1],      "data": [0],   "dtype": "int64"},
  "attention_mask":     {"shape": [1, 192], "data": [...], "dtype": "float32"},
  "future_covariates":  {"shape": [1, 16],  "data": [...], "dtype": "float32"},
  "num_output_patches": {"shape": [],       "data": [1],   "dtype": "int64"}
}
```

The `context_length` (192 in the example) must be a multiple of 16.
Use `make_input.py` or equivalent preprocessing before publishing.

---

## File inventory

| File                  | Committed | Description                              |
|-----------------------|-----------|------------------------------------------|
| `download_model.sh`   | yes       | Downloads model.onnx and runs fix        |
| `fix_model.py`        | yes       | Graph patch: adds Cast node for indices  |
| `make_input.py`       | yes       | CSV → input JSON converter (stdlib only) |
| `README.md`           | yes       | This file                                |
| `model.onnx`          | no        | Downloaded from HuggingFace (~456 MB)    |
| `model_fixed.onnx`    | no        | Patched model (same size, gitignored)    |
| `chronos_input.json`  | no        | Generated from chicken.csv (gitignored)  |
