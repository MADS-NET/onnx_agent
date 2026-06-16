# Project Purpose

To provide a MADS Agent that feeds input data to an ONNX model and publishes the results to a topic. This is intended to be as flexible as possible, so that it can be used with any ONNX model and any data source. 

Input data may be internally generated (e.g. by reading from a device or an API) or externally provided (e.g. by another MADS Agent). The ONNX model may be used for any purpose, such as classification, regression, anomaly detection, etc. The output data is to be consumed by other MADS Agents for further processing, visualization, or actuation.

As such, the model may work as a *filter* agent, when the data is provided by another agent, or as a *source* agent, when it reads data from an internal source.

# Requirements

* We provide two agents: 
  * `mads-onnx-source`: reads data from an internal source, runs the ONNX model, and publishes the output to a topic. 
  * `mads-onnx-filter`: subscribes to a topic, runs the ONNX model on the input data, and publishes the results to another topic. 
* The agents must be able to load an ONNX model from a configurable file path, defined at runtime via `mads.ini` or by override CLI option (`--model-path`).
* The agents must have a `--inspect` option that prints the model's input and output tensor names, shapes, and data types, and an example schema of the expected JSON input for the `mads-onnx-filter` agent.

# Building

* Must be portable on Linux (x86_64 and ARM64), Windows, and macOS.
* Use CMake, C++20, and modern C++ best practices. Follow the existing MADS codebase conventions for structure, naming, and style.
* Project version is read from last git tag in the format `v1.2.3`, which can be overridden by setting a `-DVERSION` on configure, or default to `0.1.0` if no tags are found.
* CMake shall provide a `package` target for the agent, which can be installed to a system location or used as a dependency by other CMake projects. The package should export the necessary include directories and libraries for using the agent as a standalone executable.
* Use the `onnxruntime` C++ API to load and run the ONNX model. Provide it by a small `FetchContent` of the release `.tgz/.zip` with `find_library` from the official ONNX Runtime repository. Prefer static linking when possible.
* Use the `Mads::AgentApp` class as the base for the agent, which provides common functionality for connecting to the MADS broker, subscribing/publishing topics, and handling settings.
* Add a GitHub action workflow for building the agent on all supported platforms. Use CMake's cross-compilation capabilities to build for multiple architectures.

# Specifics

User-facing code must have Doxygen documentation.

Each agent must provide a `--test` option that runs a standalone (no MADS network)self-test of the agent's functionality, including loading the model, running inference on a sample input, and printing the output as a JSON string.

Default pub topics are agents names without the `mads-` prefix, and dots replaced by dashes. For example, `mads-onnx-filter` would have a default sub topic of `onnx-filter`. This can be overridden by the `sub_topic` setting in the INI file or by the `--name` CLI option (as standard in MADS).

## Tensor <-> JSON mapping

flat `{"shape": [...], "data": [...]}` per tensor, with the following rules:
- For numeric tensors, `data` is a flat array of values in row-major order.
- For string tensors, `data` is a flat array of strings.
- For boolean tensors, `data` is a flat array of `true`/`false` values.
- For complex tensors, `data` is a flat array of objects with `real` and `imag` fields.

Map JSON integers to `int32` tensors, and JSON floats to `float32` tensors by default. Support for other data types (e.g. `int64`, `float64`, etc.) can be added as needed, with appropriate type annotations in the JSON schema (`{"shape": [...], "data": [...], "dtype": "uint8"}`).


## Filter agent

The `mads-onnx-filter` agent shall be general purpose and do not require any code customization/recompilation: changing ONNX model only means to inspect it and provide input JSON adhering the proper schema. The agent shall be able to subscribe to a topic, receive JSON messages, run the ONNX model on the input data, and publish the results to another topic. The input JSON schema shall be derived from the model's input tensor names, shapes, and data types. The output JSON schema shall be derived from the model's output tensor names, shapes, and data types.

The project shall provide an example using the <https://huggingface.co/kashif/chronos-2-onnx> model, which is a time series forecasting model. The `--test` commad shall run the model on a sample input JSON and print the output JSON. 

Timing of filter agent is triggered by the input.

Sub topics are configurable via `mads.ini` or CLI options, with the same rules as the source agent. The default sub topic for the filter agent is `onnx-filter-input`, and the default pub topic is `onnx-filter`.

## Source agent

The `mads-onnx-source` agent will use a standard data source implemented as a C++ class inheriting a `Onnx::DataSource` pure-virtual interface. The data source will be responsible for generating or reading data from a device or an API, and providing it to the agent in a format compatible with the ONNX model's input tensor. The agent will then run the ONNX model on the input data and publish the results to a topic. The data source class can be customized or replaced by other implementations as needed.

The project shall provide a default data source that grabs camera frames from a webcam using OpenCV (`findPackage(OpenCV)`, system-installed), converts them to the appropriate format for the ONNX model, and provides them to the agent. The default data source shall be configurable via `mads.ini` or CLI options, such as camera index, resolution, frame rate, etc. **Compiled-out if OpenCV is absent**.

The default ONNX model for that agent will be a simple image classification model, such as MobileNetV2 or ResNet50, which can be downloaded from the ONNX Model Zoo <https://huggingface.co/onnxmodelzoo>. The model shall be provided via a download script. The `--test` command shall run the model on a sample image and print the output JSON with the predicted class and confidence score (softmax + top-k, and an ImageNet labels file (1000 classes)).

The ini file must provide model-specific image preprocessing constants.

Timing of the source agent is internal, based on the data source's update frequency (e.g. camera frame rate), or slowed down by the `--period` option (on CLI or in the INI file).