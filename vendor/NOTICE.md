# Third-party code in app/vendor

| Path | Package | Version | License |
|---|---|---|---|
| `ort/` | onnxruntime-web (Microsoft) | 1.20.1 | MIT |
| `tjs/transformers.js` | @huggingface/transformers | 4.3.2 | Apache-2.0 (see `tjs/LICENSE-transformers.js`) |
| `tjs/ort-wasm-simd-threaded.asyncify.*` | onnxruntime-web (Microsoft) | 1.31.0-dev.20260914-8d85527a0 | MIT |

ONNX Runtime is Copyright (c) Microsoft Corporation, MIT License: https://github.com/microsoft/onnxruntime/blob/main/LICENSE
Files are unmodified copies from the npm packages, vendored so the app works fully offline. The
non-minified transformers.js build is used because GitHub secret scanning flags a false positive in the minified one.
