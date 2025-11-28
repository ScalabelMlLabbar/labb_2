import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# Configuration
MODEL_REPO = "SaiBon99/llama-finetuned-gguf"
MODEL_FILE = "model.gguf"

print(f"Downloading model from {MODEL_REPO}...")

try:
    # Download the GGUF model file from HuggingFace
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        repo_type="model"
    )
    print(f"Model downloaded to: {model_path}")

    # Load the model with llama-cpp-python
    print("Loading model into memory...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window
        n_threads=4,  # Number of CPU threads
        n_gpu_layers=0,  # Set to 0 for CPU, increase for GPU
        verbose=False,
    )
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    llm = None

def chat(message, history):
    """
    Chat function that takes a message and chat history,
    and returns the model's response.
    """
    if llm is None:
        return "Error: Model failed to load. Please check the logs."

    # Build the conversation in Llama 3 chat format
    conversation = "<|begin_of_text|>"

    if history:
        for user_msg, bot_msg in history:
            conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            conversation += f"<|start_header_id|>assistant<|end_header_id|>\n\n{bot_msg}<|eot_id|>"

    # Add the current message
    conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
    conversation += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    # Generate response
    try:
        response = llm(
            conversation,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|eot_id|>", "<|start_header_id|>"],
            echo=False
        )

        return response['choices'][0]['text'].strip()

    except Exception as e:
        return f"Error generating response: {str(e)}"

# Create Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat,
    title="Llama 3.2 3B Fine-tuned Chat",
    description=f"Chat with Llama 3.2 3B fine-tuned on FineTome-100k dataset (GGUF format)\n\nModel: `{MODEL_REPO}`",
    examples=[
        "Explain what boolean operators are and how they work in programming.",
        "What is the difference between short-circuit evaluation and normal evaluation?",
        "Write a Python function to check if a number is prime.",
        "Explain the concept of operator precedence with examples.",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
