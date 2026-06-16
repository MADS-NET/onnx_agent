# ImageNet Classification Example

This example uses **MobileNetV2-12** (from the ONNX Model Zoo) with
`mads-onnx-source` to classify webcam frames in real time against the
ImageNet 1000-class taxonomy.

## Requirements

- OpenCV 4.x installed on the build host (`find_package(OpenCV)` must succeed).
- The `mads-onnx-source` binary built with OpenCV support (check the CMake
  configure output for `OpenCV found: …`).

## 1. Download the model and assets

```bash
bash examples/imagenet/download_model.sh
```

This downloads:

| File | Source | Purpose |
|------|--------|---------|
| `mobilenetv2-12.onnx` (~13 MB) | ONNX Model Zoo (GitHub) | MobileNetV2 classifier |
| `imagenet_classes.txt` | PyTorch Hub | 1000 ImageNet class labels |
| `sample_cat.jpg` | COCO val 2017 | Test image (two cats) |

> The `.onnx` model and `sample_cat.jpg` are listed in `.gitignore` and are
> not committed to the repository.  The labels file (`imagenet_classes.txt`)
> _is_ committed because it is small (~10 KB) and static.

## 2. Inspect the model

```bash
./build/src/main/mads-onnx-source \
  --inspect \
  -m examples/imagenet/mobilenetv2-12.onnx
```

Expected output:
```
Model: examples/imagenet/mobilenetv2-12.onnx

Inputs:
  input
    dtype : float32
    shape : [?, 3, 224, 224]

Outputs:
  output
    dtype : float32
    shape : [?, 1000]
```

The input is NCHW float32 (batch × 3 channels × 224 × 224 pixels).

## 3. Smoke-test (no MADS broker)

```bash
./build/src/main/mads-onnx-source \
  --test \
  -m examples/imagenet/mobilenetv2-12.onnx \
  --image examples/imagenet/sample_cat.jpg \
  --labels examples/imagenet/imagenet_classes.txt
```

Expected output (abridged):
```json
{
  "best": {
    "class_id": 281,
    "confidence": 0.7495,
    "label": "tabby"
  },
  "top_k": [
    { "class_id": 281, "label": "tabby",        "confidence": 0.7495 },
    { "class_id": 285, "label": "Egyptian cat",  "confidence": 0.1142 },
    { "class_id": 282, "label": "tiger cat",     "confidence": 0.1081 },
    ...
  ]
}
```

A plausible top prediction (tabby/Egyptian cat/tiger cat) confirms that the
preprocessing pipeline (resize → BGR→RGB → [0,1] scaling → ImageNet
mean/std normalisation → NCHW float32 tensor) is correct.

## 4. Live webcam (with MADS broker)

Configure `mads.ini` (the project root already contains a ready-made
`[onnx-source]` section):

```ini
[onnx-source]
model_path   = "examples/imagenet/mobilenetv2-12.onnx"
labels_path  = "examples/imagenet/imagenet_classes.txt"
pub_topic    = "onnx-source"
period       = 100        # ms between frames (~10 fps)
top_k        = 5
camera       = 0
```

Start the agent:

```bash
./build/src/main/mads-onnx-source \
  --settings mads.ini \
  -m examples/imagenet/mobilenetv2-12.onnx \
  --labels examples/imagenet/imagenet_classes.txt
```

Each published message on topic `onnx-source` has the format:

```json
{
  "best":  { "class_id": 281, "label": "tabby", "confidence": 0.75 },
  "top_k": [...]
}
```

## Pre-processing constants

| Parameter | Value | Notes |
|-----------|-------|-------|
| Input size | 224 × 224 | Standard ImageNet |
| Layout | NCHW | Batch × Channel × H × W |
| Channel order | RGB | Convert from OpenCV BGR |
| Scale | 1/255 | Maps [0,255] → [0,1] |
| Mean | [0.485, 0.456, 0.406] | Per-channel ImageNet mean |
| Std | [0.229, 0.224, 0.225] | Per-channel ImageNet std |

These match the standard torchvision ImageNet preprocessing and are verified
against the MobileNetV2-12 ONNX Model Zoo checkpoint.
