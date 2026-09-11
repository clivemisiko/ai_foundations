# AI Foundations — Gate H1: AI Landscape, LLMs & Tooling

**Author:** Misiko  
**Track:** Track H — AI Foundations (Gate 22 of 39)  
**Status:** Completed & Verified  

---

## 📌 Overview

This repository represents the deliverables for **Gate H1: AI Landscape, LLMs & Tooling**. It establishes foundational competencies in navigating the modern Artificial Intelligence ecosystem, verifying essential AI developer tooling, and engineering reliable LLM integrations using secure environment practices and strictly typed structured outputs (JSON schema enforcement via Pydantic).

---

## 📂 Repository Structure

```text
ai_foundations_h1/
│
├── AI_LANDSCAPE_MAP.md            # Comprehensive taxonomy of AI subfields, paradigms, tools & 12 case studies
├── structured_llm_response.ipynb  # Interactive notebook: Calling Gemini API with Pydantic structured output
├── verify_env.py                  # Health check & connectivity verification script (HF, Kaggle, Gemini)
├── .env                           # Environment secrets (GEMINI_API_KEY) — excluded via .gitignore
├── .gitignore                     # Git ignore rules protecting credentials and virtual environments
└── README.md                      # Project documentation and summary
```

---

## 🚀 Key Deliverables & What We Built

### 1. AI Landscape Map (`AI_LANDSCAPE_MAP.md`)
An in-depth conceptual breakdown and industry reference map spanning the five core disciplines of modern AI:

* **Machine Learning (ML):** Statistical models learning decision boundaries from structured/tabular data (scikit-learn, XGBoost, LightGBM).
* **Deep Learning (DL):** Multi-layered artificial neural networks automatically learning representations from high-dimensional unstructured data (PyTorch, TensorFlow/Keras, JAX, CUDA).
* **Natural Language Processing (NLP):** Linguistic understanding, syntactic parsing, and semantic embedding pipelines (Hugging Face Transformers, spaCy, tiktoken, vector databases).
* **Computer Vision (CV):** Visual perception, object detection, and spatial segmentation (OpenCV, torchvision, YOLO, SAM 2).
* **Generative AI (GenAI):** Foundation models synthesizing novel text, multimodal artifacts, and code (Google GenAI SDK, frontier LLMs, Diffusion, RAG).
* **12 Named Production Case Studies:** Real-world analyses of Stripe Radar, Netflix Recommendations, Zillow Zestimate, DeepMind AlphaFold, Tesla FSD, Google Translate, BloombergGPT, Apple Face ID, Waymo Driver, Gemini Code Assist, Midjourney, and Perplexity AI.

---

### 2. Structured LLM Response Pipeline (`structured_llm_response.ipynb`)
An end-to-end Python / Jupyter implementation demonstrating production best practices for calling Large Language Models:

* **Zero-Leak Credential Management:** Reads credentials dynamically via `python-dotenv` from `.env`, masking sensitive tokens in console outputs and keeping keys strictly outside source code and commits.
* **Modern SDK Integration:** Leverages the official modern `google-genai` SDK (`genai.Client`) with `gemini-3.6-flash`.
* **Strict Type Safety & Data Contract:** Uses **Pydantic v2** (`BaseModel`, `Field`) to define `AIConceptAnalysis`:
  ```python
  class AIConceptAnalysis(BaseModel):
      subfield: str = Field(description="The primary AI subfield (ML, DL, NLP, CV, or GenAI)")
      concept_name: str = Field(description="Name of the concept or technology")
      summary: str = Field(description="A concise technical summary of how it works")
      primary_tool: str = Field(description="The dominant framework or library used (e.g., PyTorch, Hugging Face)")
      real_world_example: str = Field(description="A named real-world production use case")
      key_takeaway: str = Field(description="One sentence summary of its core significance")
  ```
* **Native Structured Output Configuration:** Passes the Pydantic class directly into `config={"response_mime_type": "application/json", "response_schema": AIConceptAnalysis}`, forcing the model to adhere to the schema at decode-time.
* **Schema Validation & Parsing:** Deserializes and validates model responses using `AIConceptAnalysis.model_validate_json(response.text)`.

---

### 3. Platform & Environment Verification (`verify_env.py`)
An automated diagnostic test script validating local and cloud AI toolchains:
* **Python runtime version** (3.12+).
* **JupyterLab installation** and kernel environment readiness.
* **Hugging Face Hub API connectivity** (queried `google/gemma-2-2b`).
* **Kaggle tooling integration** and CLI availability.
* **Google Gemini API authentication** and live test inference (`ENVIRONMENT_READY`).

---

## 🛠️ Setup & Usage Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 1. Environment Setup
Clone or navigate to the project root and activate your virtual environment:

```powershell
# Create virtual environment (if not already created)
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

Install required packages:
```powershell
pip install python-dotenv pydantic google-genai jupyterlab ipykernel huggingface_hub kaggle
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Verify the Environment
Run the verification script:
```powershell
python verify_env.py
```
Expected output:
```text
=== Environment & Platform Verification ===
1. Python Version: 3.12.0
2. JupyterLab: Ready
3. Hugging Face Access: Verified
4. Kaggle Tooling: Verified
5. API Key in Env: Loaded securely (AQ.A...-vNQ) - not hardcoded
6. Gemini API Call: Success! Response = ENVIRONMENT_READY
============================================
```

### 4. Running the Notebook
Launch JupyterLab or open the workspace in VS Code:
```powershell
jupyter lab
```
Open `structured_llm_response.ipynb`. In the kernel dropdown (top-right), select **`Python (.venv)`** and click **Run All**.

