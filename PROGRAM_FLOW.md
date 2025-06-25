# Program Flow

This document outlines how `shapeshifter.py` operates and the key objects involved.

```mermaid
flowchart TD
    A[Start] --> B[build_gradio_app()]
    B --> C{gradio installed?}
    C -- no --> Z[ImportError]
    C -- yes --> D[Create Blocks]
    D --> E["Layer Transformation Demo"]
    D --> F["Memory & Inference"]
    E --> G[run_wrap_demo]
    G --> H[wrap_demo(height,width)]
    H --> I[TransversalWrapLayer.call]
    F --> J[infer_button.click]
    J --> K[check_memory_and_infer(prompt)]
    K --> L{enough memory?}
    L -- yes --> M[Load T5 model & generate]
    L -- no --> N[Return warning]
    M --> O[Return generated text]
    N --> O
    E --> P[Display before/after]
    O --> Q[Display output]
    B --> R[Return demo]
    R --> S[demo.launch()]
```

## Overview
- **TransversalWrapLayer** — custom Keras layer to produce seamless wrapping of 2D tensors.
- **wrap_demo** — creates a numeric grid and applies the wrapping layer to show before/after results.
- **check_memory_and_infer** — verifies system memory and optionally runs a T5 model for text generation.
- **build_gradio_app** — assembles the Gradio UI with two tabs that call the above utilities.

`shapeshifter.py` runs `build_gradio_app()` and launches the Gradio interface when executed as a script.
