# LLM_app1
It is an intelligent web application that leverages Microsoft's Phi-2 language model to help developers generate and analyze code across multiple programming languages. 

# Features
1. Code Generation

Multi-Language Support: Generate code in Python, JavaScript, Java, HTML/CSS, React, and Node.js
Detailed Output: Receive complete code with:

Working implementation
Code explanation
Assumptions made
Usage examples

2. Code Analysis

Comprehensive Review: Analyze code from multiple perspectives
Analysis Types:

Optimization
Security
Style
Correction

# Prerequisites

Python 3.8+
pip package manager
Minimum 8GB RAM recommended
Stable internet connection for model download

# How It Works

Model Loading: Automatically loads Microsoft's Phi-2 model
Prompt Engineering: Crafts intelligent prompts for code generation
Intelligent Generation: Uses advanced sampling techniques

Temperature-based randomness
Nucleus sampling
Top-k token selection

# Generation Parameters

Temperature: 0.7 (Balanced creativity)
Max Length: 2048 tokens
Sampling Method: Top-p (Nucleus) sampling

# Limitations

First load may take several minutes
Runs on CPU (slower processing)
Generated code might require manual review
Limited to 2048 token outputs

# Experience it!!

You can use the below link to experience the project
https://huggingface.co/spaces/VishakBaddur/LLM_app

Note: It will take time to run(approx 20mins), because it is running on the basic free CPU version.
Please be patient.
