import streamlit as st
import os
from typing import Dict, List, Optional, Tuple
from llama_cpp import Llama
import wget
import time

def download_model_if_needed():
    """Download the model if it's not present and show progress."""
    model_dir = "models"
    model_path = os.path.join(model_dir, "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    
    if not os.path.exists(model_path):
        try:
            st.info("Downloading model... This may take a few minutes.")
            progress_bar = st.progress(0)
            os.makedirs(model_dir, exist_ok=True)
            
            model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            
            # Custom hook for download progress
            def download_progress(current, total, width=80):
                progress = float(current) / float(total)
                progress_bar.progress(progress)
            
            wget.download(model_url, model_path, bar=download_progress)
            progress_bar.progress(1.0)
            st.success("Model downloaded successfully!")
            
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            return None
            
    return model_path

class InteractiveCodeTools:
    def __init__(self, model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"):
        """Initialize the CodeTools with local Mistral model."""
        self.model_path = model_path
        
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=8192,
                n_threads=4,
                n_gpu_layers=0
            )
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            raise e
        
        self.supported_languages = {
            "Python": "Python",
            "JavaScript": "JavaScript",
            "Java": "Java",
            "HTML/CSS": "HTML/CSS",
            "React": "React",
            "Node.js": "Node.js"
        }

    def generate_code(self, language: str, prompt: str) -> str:
        """Generate code based on user requirements."""
        generation_prompt = f"""<s>[INST] As an expert {language} developer, please generate code based on the following requirements:

{prompt}

Please provide:
1. Complete, working code that fulfills the requirements
2. Brief explanation of how the code works
3. Any assumptions made
4. Usage examples

Make sure the code follows best practices, includes proper error handling, and is well-documented. [/INST]"""

        try:
            result = self.query_model(generation_prompt, max_tokens=2000)
            return result
        except Exception as e:
            return f"Error during code generation: {str(e)}"

    def analyze_code(self, language: str, code: str, analysis_type: str = "all") -> Dict:
        """Analyze code based on user preferences."""
        prompts = {
            "optimization": f"""You are an expert {language} developer. Analyze this code for performance optimization opportunities. Consider:
                1. Time complexity
                2. Space complexity
                3. Algorithm efficiency
                4. Resource usage
                Provide specific suggestions with examples.""",
            
            "security": f"""You are an expert security engineer. Analyze this {language} code for security vulnerabilities. Consider:
                1. Input validation
                2. Data sanitization
                3. Common security pitfalls
                4. Best security practices
                Provide specific fixes with examples.""",
            
            "style": f"""You are an expert {language} developer. Review this code for style improvements. Consider:
                1. Language-specific conventions
                2. Code organization
                3. Documentation
                4. Naming conventions
                Provide specific suggestions with examples.""",
                
            "correction": f"""You are an expert {language} developer. Review this code for errors and potential bugs. Consider:
                1. Syntax errors
                2. Logical errors
                3. Edge cases
                4. Exception handling
                Provide the corrected code with explanations of the changes made."""
        }

        results = {}
        analysis_types = ["optimization", "security", "style", "correction"] if analysis_type == "all" else [analysis_type]

        for current_type in analysis_types:
            prompt = f"""<s>[INST] As an expert code reviewer, please analyze this {language} code:

{code}

{prompts[current_type]}

Format your response as:
1. Summary of findings
2. Detailed suggestions (with code examples)
3. Priority level for each suggestion (High/Medium/Low)
{'4. Complete corrected code' if current_type == 'correction' else ''} [/INST]"""

            try:
                results[current_type] = self.query_model(prompt)
            except Exception as e:
                results[current_type] = f"Error during {current_type} analysis: {str(e)}"

        return results

    def query_model(self, prompt: str, max_tokens: int = 1500) -> str:
        """Query the local Mistral model."""
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["</s>", "[/INST]"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error querying model: {str(e)}"

# Streamlit interface
st.set_page_config(page_title="Code Generator & Analyzer", layout="wide")
st.title("Code Generator & Analyzer")

# Initialize the tool in session state if it doesn't exist
if 'tool' not in st.session_state:
    with st.spinner("Initializing the application..."):
        model_path = download_model_if_needed()
        if model_path:
            try:
                st.session_state.tool = InteractiveCodeTools(model_path)
            except Exception as e:
                st.error("Failed to initialize the application. Please try again later.")
                st.stop()

# Create sidebar for main operation choice
choice = st.sidebar.radio(
    "What would you like to do?",
    ["Generate new code from prompt", "Analyze existing code"]
)

if choice == "Generate new code from prompt":
    # Language selection
    language = st.sidebar.selectbox(
        "Choose a programming language",
        list(st.session_state.tool.supported_languages.keys())
    )
    
    # Get user prompt
    prompt = st.text_area(
        "Enter your requirements/prompt for code generation:",
        height=200,
        placeholder="Describe what you want the code to do, any specific features or constraints..."
    )
    
    if st.button("Generate Code"):
        if prompt:
            with st.spinner("Generating code..."):
                try:
                    results = st.session_state.tool.generate_code(language, prompt)
                    st.markdown("### Generated Code and Documentation")
                    st.markdown(results)
                except Exception as e:
                    st.error(f"Error generating code: {str(e)}")
        else:
            st.warning("Please enter a prompt for code generation.")

else:  # Analyze existing code
    # Language selection
    language = st.sidebar.selectbox(
        "Choose a programming language",
        list(st.session_state.tool.supported_languages.keys())
    )
    
    # Code input
    code = st.text_area(
        "Enter your code:",
        height=300,
        placeholder="Paste your code here..."
    )
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Choose analysis type",
        ["all", "optimization", "security", "style", "correction"],
        format_func=lambda x: x.title()
    )
    
    if st.button("Analyze Code"):
        if code:
            with st.spinner("Analyzing code..."):
                try:
                    results = st.session_state.tool.analyze_code(language, code, analysis_type)
                    for analysis_key, analysis_result in results.items():
                        with st.expander(f"{analysis_key.title()} Analysis", expanded=True):
                            st.markdown(analysis_result)
                except Exception as e:
                    st.error(f"Error analyzing code: {str(e)}")
        else:
            st.warning("Please enter some code to analyze.")
