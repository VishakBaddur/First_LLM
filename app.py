import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from threading import Lock
import warnings
warnings.filterwarnings('ignore')

# Global lock for model inference
inference_lock = Lock()

def load_model():
    print("Starting model load...")
    # Using Microsoft's Phi-2 model which is open access
    model_name = "microsoft/phi-2"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully!")
    return model, tokenizer

def format_prompt(instruction):
    """Format prompt for the model."""
    return f"""Instruct: {instruction}\nOutput:"""

def generate_code(language: str, prompt: str, model, tokenizer):
    """Generate code based on user requirements."""
    generation_prompt = format_prompt(
        f"""As an expert {language} developer, please generate code based on the following requirements:

{prompt}

Please provide:
1. Complete, working code that fulfills the requirements
2. Brief explanation of how the code works
3. Any assumptions made
4. Usage examples"""
    )

    with inference_lock:  # Ensure only one generation at a time
        try:
            inputs = tokenizer(generation_prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_length=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error generating code: {str(e)}\nTry refreshing the page if the model failed to load."

def analyze_code(language: str, code: str, analysis_type: str, model, tokenizer):
    """Analyze code based on user preferences."""
    analysis_prompt = format_prompt(
        f"""As an expert code reviewer, analyze this {language} code:

{code}

Focus on {analysis_type} aspects and provide:
1. Summary of findings
2. Detailed suggestions with examples
3. Priority level for each suggestion"""
    )

    with inference_lock:
        try:
            inputs = tokenizer(analysis_prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_length=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error analyzing code: {str(e)}\nTry refreshing the page if the model failed to load."

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Code Generator & Analyzer")
    
    # Create a loading message
    loading_msg = gr.Markdown("Loading model... This may take a few minutes on first load.")
    
    # Load model
    try:
        model, tokenizer = load_model()
        loading_msg.value = "Model loaded successfully! Ready to use."
    except Exception as e:
        loading_msg.value = f"Error loading model: {str(e)}\nTry refreshing the page."
    
    with gr.Tab("Generate Code"):
        with gr.Row():
            language_input = gr.Dropdown(
                choices=["Python", "JavaScript", "Java", "HTML/CSS", "React", "Node.js"],
                label="Programming Language",
                value="Python"
            )
            prompt_input = gr.Textbox(
                lines=5,
                label="Requirements/Prompt",
                placeholder="Describe what you want the code to do..."
            )
        generate_button = gr.Button("Generate Code")
        code_output = gr.Markdown(label="Generated Code")
        
        generate_button.click(
            fn=lambda l, p: generate_code(l, p, model, tokenizer),
            inputs=[language_input, prompt_input],
            outputs=code_output
        )
    
    with gr.Tab("Analyze Code"):
        with gr.Row():
            code_input = gr.Textbox(
                lines=10,
                label="Code to Analyze",
                placeholder="Paste your code here..."
            )
            analysis_type = gr.Dropdown(
                choices=["optimization", "security", "style", "correction"],
                label="Analysis Type",
                value="style"
            )
        analyze_button = gr.Button("Analyze Code")
        analysis_output = gr.Markdown(label="Analysis Results")
        
        analyze_button.click(
            fn=lambda c, l, t: analyze_code(l, c, t, model, tokenizer),
            inputs=[code_input, language_input, analysis_type],
            outputs=analysis_output
        )

    gr.Markdown("""
    ### Notes:
    - First load may take a few minutes as the model is downloaded and initialized
    - If you get an error, try refreshing the page
    - Generation may take 30-60 seconds due to CPU processing
    """)

demo.launch()
