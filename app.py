import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from loguru import logger

# Configuration
# Using an open-access model that doesn't require authentication
# Options:
# 1. "HuggingFaceTB/SmolLM2-1.7B-Instruct" - Small, fast, no auth required
# 2. "meta-llama/Llama-3.2-3B-Instruct" - Better quality but requires HF login
# 3. "mistralai/Mistral-7B-Instruct-v0.3" - Larger, better quality
MODEL_REPO = "igobl/Llama-3.2-1B-Instruct-bnb-4bit"

logger.info(f"Loading model from {MODEL_REPO}...")

try:
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    logger.success(f"Tokenizer loaded successfully (vocab size: {tokenizer.vocab_size})")

    # Load the model for CPU inference
    logger.info("Loading model into memory (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        dtype=torch.float32,  # Use float32 for CPU, or bfloat16 to save memory
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # Set to evaluation mode
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    logger.success(f"Model loaded successfully! Parameters: {param_count:,}")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.exception("Full traceback:")
    model = None
    tokenizer = None

def chat(message, history):
    """
    Chat function that takes a message and chat history,
    and returns the model's response.
    """
    if model is None or tokenizer is None:
        logger.error("No model or tokenizer loaded!")
        return "Error: Model failed to load. Please check the logs."

    logger.info(f"Received message: {message[:100]}..." if len(message) > 100 else f"Received message: {message}")

    # Build the conversation using the chat template
    messages = []

    # Add history
    if history:
        logger.debug(f"Chat history has {len(history)} previous messages")
        logger.info(f'history: {history}')

        # Handle the new Gradio format where history is a list of message dictionaries
        for msg in history:
            role = msg.get('role', 'user')

            # Extract text content from the message
            content = msg.get('content', [])
            if isinstance(content, list):
                # Content is a list of content items, extract text from the first text item
                content_text = ""
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content_text = item.get('text', '')
                        break
            else:
                # Fallback if content is already a string
                content_text = content

            messages.append({"role": role, "content": content_text})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Generate response
    try:
        # Apply chat template and tokenize
        logger.debug("Applying chat template...")
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        input_length = inputs['input_ids'].shape[1]
        logger.info(f"Input tokens: {input_length}")

        # Generate
        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        output_length = outputs.shape[1]
        tokens_generated = output_length - input_length
        logger.info(f"Generated {tokens_generated} new tokens")

        response = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        logger.success(f"Response generated successfully: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
        return response.strip()

    except Exception as ex:
        logger.error(f"Error generating response: {str(ex)}")
        logger.exception("Full traceback:")
        return f"Error generating response: {str(ex)}"

# Create Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat,
    title="Llama 3.2 1B Fine-tuned with Fine Tome Chat",
    description=f"Chat with Llama 3.2 1B using Hugging Face Transformers (CPU inference)\n\nModel: `{MODEL_REPO}`",
    examples=[
        "Explain what boolean operators are and how they work in programming.",
        "What is the difference between short-circuit evaluation and normal evaluation?",
        "Write a Python function to check if a number is prime.",
        "Explain the concept of operator precedence with examples.",
    ],
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(server_name="0.0.0.0")  # Let Gradio auto-select available port
    logger.info("Gradio interface launched successfully")
