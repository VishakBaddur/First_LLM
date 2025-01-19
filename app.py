import streamlit as st
import os
from typing import Dict, List, Optional, Tuple
from llama_cpp import Llama

[Paste your entire InteractiveCodeTools class here, from the __init__ method to the format_results method]

# Streamlit interface
st.set_page_config(page_title="Code Generator & Analyzer", layout="wide")
st.title("Code Generator & Analyzer")

# Initialize the tool in session state if it doesn't exist
if 'tool' not in st.session_state:
    model_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    st.session_state.tool = InteractiveCodeTools(model_path)

# Create sidebar for main operation choice
choice = st.sidebar.radio(
    "What would you like to do?",
    ["Generate new code from prompt", "Analyze existing code"]
)

if choice == "Generate new code from prompt":
    # Language selection
    language = st.sidebar.selectbox(
        "Choose a programming language",
        list(st.session_state.tool.supported_languages.values())
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
                results = st.session_state.tool.generate_code(language, prompt)
                st.markdown("### Generated Code and Documentation")
                st.markdown(results)
        else:
            st.warning("Please enter a prompt for code generation.")

else:  # Analyze existing code
    # Language selection
    language = st.sidebar.selectbox(
        "Choose a programming language",
        list(st.session_state.tool.supported_languages.values())
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
                results = st.session_state.tool.analyze_code(language, code, analysis_type)
                
                # Display formatted results
                for analysis_key, analysis_result in results.items():
                    with st.expander(f"{analysis_key.title()} Analysis", expanded=True):
                        st.markdown(analysis_result)
        else:
            st.warning("Please enter some code to analyze.")
