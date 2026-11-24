# ChexLite note model

Fine-tuned from **Gemma 3 270M (instruction-tuned)** by Google (LoRA SFT + GRPO, merged, 4-bit/8-bit ONNX).

Gemma is provided under and subject to the Gemma Terms of Use found at https://ai.google.dev/gemma/terms.
Use of this model must comply with the Gemma Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy.

The model rewrites the ChexLite classifier's structured output as a short note. It does not see images,
adds no diagnostic information, and every note is checked by a deterministic verifier before display.
It is not a medical device.
