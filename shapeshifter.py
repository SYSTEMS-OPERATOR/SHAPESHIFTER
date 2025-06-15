"""Shapeshifter demo module.

This script showcases a custom Keras layer that performs a toroidal wrap on
2D tensors and provides a Gradio interface for quick experimentation. The demo
also includes a lightweight text-generation example to verify memory
availability before running a small T5 model.
"""

import psutil
import gradio as gr
import tensorflow as tf
from tensorflow.keras.layers import Layer
import transformers

##############################################
# 1) Custom TransversalWrapLayer
##############################################

class TransversalWrapLayer(Layer):
    """
    A custom Keras layer that implements seamless wrapping of 2D tensors.
    By default, it pads the input with one row/col from the opposite edge 
    to create a "toroidal" wrap effect.
    """

    def __init__(self, **kwargs):
        super(TransversalWrapLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Transverses the tensor to create seamless edges by wrapping rows and columns.

        Expected shape: (batch_size, height, width, channels).
        Returns: (batch_size, height + 2, width + 2, channels).
        """
        # Wrap horizontally by concatenating the last column at the front and
        # the first column at the end
        wrapped = tf.concat([inputs[:, :, -1:], inputs, inputs[:, :, :1]], axis=2)
        # Wrap vertically by concatenating the last row on top and the first row
        # at the bottom
        wrapped = tf.concat([wrapped[:, -1:, :], wrapped, wrapped[:, :1, :]], axis=1)
        return wrapped

##############################################
# 2) Simple "Shapeshift" Demo
#    - We'll show how to apply the custom wrap 
#      layer to an input tensor.
##############################################
def wrap_demo(height, width):
    """
    Generates a dummy tensor (with shape (1, height, width, 1))
    to demonstrate TransversalWrapLayer. Returns a preview before and after.
    """
    # Make a simple test input: a range of ints reshaped to (1,H,W,1)
    # so we can visualize how the wrap changes the corners.
    import numpy as np

    # create sequential data (for visualization)
    data_np = np.arange(height * width).reshape((1, height, width, 1)).astype(np.float32)
    data = tf.convert_to_tensor(data_np)

    # Apply the wrap layer
    wrap_layer = TransversalWrapLayer()
    wrapped = wrap_layer(data)

    return data_np.squeeze(axis=0).squeeze(axis=-1), wrapped.numpy().squeeze(axis=0).squeeze(axis=-1)

##############################################
# 3) Example Inference with a TF Model
#    Checking memory first, then loading T5 
#    or any small HF TF model for text generation.
##############################################
def check_memory_and_infer(prompt):
    """Run a memory check and optionally perform text generation.

    This function verifies that at least ~2 GB of RAM is free before loading
    a small TensorFlow T5 model for demonstration purposes. If the environment
    does not have enough memory, it returns a warning string instead of running
    the model.
    """

    # Check available memory
    total_mem_gb = psutil.virtual_memory().total / (1024**3)
    avail_mem_gb = psutil.virtual_memory().available / (1024**3)

    if avail_mem_gb < 2.0:
        return f"Insufficient memory ({avail_mem_gb:.2f} GB available). Try a smaller model or a bigger runtime."

    # If we have enough memory, load a small T5 model from HF (TensorFlow version)
    # We do it once for demonstration, but ideally you'd cache or load in initialization.
    model_name = "google/t5-small-ssm-nq"  # small T5 variant with TF weights
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    t5_model = transformers.TFT5ForConditionalGeneration.from_pretrained(model_name)

    # Prepare prompt for T5: we typically prefix with "translate English to French:" or
    # "summarize: " or whatever task. For a naive text generation, we can do a simple approach:
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    outputs = t5_model.generate(input_ids, max_length=50, num_beams=2, no_repeat_ngram_size=2)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"**Memory Check**:\n- Total: {total_mem_gb:.2f} GB\n- Available: {avail_mem_gb:.2f} GB\n\n**Generated Text**:\n{decoded}"

##############################################
# 4) Build Gradio Interface
##############################################
def build_gradio_app():
    """Construct and return the Gradio Blocks demo."""

    with gr.Blocks() as demo:

        gr.Markdown("# Shapeshifter Mini-IDE")
        gr.Markdown(
            "Welcome to the Shapeshifter environment! Below you can:\n"
            "1. **Seamlessly wrap** 2D tensors with TransversalWrapLayer.\n"
            "2. **Perform text-generation** in a second tab if memory allows.\n"
            "\nThis demo is designed for Kaggle, Colab, or local usage. "
            "No complicated setup required!"
        )

        with gr.Tab("Layer Transformation Demo"):
            gr.Markdown(
                "### Transversal Wrap Demo\n"
                "Enter the size of a dummy 2D tensor. We'll show you the original data and the wrapped version."
            )
            height_input = gr.Number(label="Height", value=4, precision=0)
            width_input = gr.Number(label="Width", value=5, precision=0)
            run_button = gr.Button("Run Wrap")

            before_output = gr.Textbox(label="Original (2D slice)", interactive=False)
            after_output = gr.Textbox(label="Wrapped (2D slice)", interactive=False)

            # Define the callback for the wrap demo
            def run_wrap_demo(h, w):
                """Execute the wrapping demo and format results for display."""
                before_np, after_np = wrap_demo(int(h), int(w))
                return (str(before_np), str(after_np))

            run_button.click(
                run_wrap_demo,
                inputs=[height_input, width_input],
                outputs=[before_output, after_output],
            )

        with gr.Tab("Memory & Inference"):
            gr.Markdown(
                "### Memory Check and Text Generation\n"
                "We'll check your environment's available memory, then load a small T5 model for inference if possible."
            )
            prompt_input = gr.Textbox(
                label="Enter prompt",
                placeholder="Type something (e.g. 'Translate this sentence to French' or 'Explain gravity').",
            )
            infer_button = gr.Button("Run Inference")

            inference_output = gr.Markdown()

            infer_button.click(
                fn=check_memory_and_infer,
                inputs=prompt_input,
                outputs=inference_output
            )

    return demo


##############################################
# 5) Launch the App
##############################################
if __name__ == "__main__":
    # Just launch the Blocks directly
    demo_app = build_gradio_app()
    demo_app.launch()
