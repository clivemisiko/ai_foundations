# Track H: AI Foundations (Gates H1, H2 & H3)

**Author:** Misiko  
**Track:** Track H — AI Foundations (Gates 22, 23 & 24 of 39)  
**Status:** Completed & Verified  

---

## 📌 Repository Overview

This repository contains the verified deliverables, codebases, pipelines, and technical documentation for **Track H: AI Foundations**:
- **Gate H1: AI Landscape, LLMs & Tooling** — Comprehensive AI landscape taxonomy, developer tooling verification, and a typed, structured LLM response pipeline using the Gemini API and Pydantic.
- **Gate H2: Python for ML & First Pipeline** — An end-to-end, data-leakage-free machine learning pipeline on a real-world Credit Risk dataset using Pandas, Scikit-Learn (`ColumnTransformer`, `Pipeline`), and Random Forest.
- **Gate H3: Math Behind ML** — Pure NumPy from-scratch implementations of Gradient Descent and a 2-Layer Neural Network with analytical backpropagation (zero autograd/frameworks), numerically validated against Scikit-Learn benchmarks.

---

## 📂 Repository Structure

```text
ai_foundations_h1/
│
├── .vscode/
│   └── settings.json              # Locks default Python interpreter to local .venv
│
├── AI_LANDSCAPE_MAP.md            # Comprehensive taxonomy of 5 AI subfields & 12 production case studies
├── structured_llm_response.ipynb  # Gate H1: Calling Gemini API with Pydantic structured output
├── verify_env.py                  # Gate H1: Diagnostic test script (Hugging Face, Kaggle, Gemini)
│
├── ml_first_pipeline.ipynb        # Gate H2: End-to-end ML pipeline on Credit Risk dataset
├── math_behind_ml.ipynb           # Gate H3: Gradient Descent & 2-Layer Neural Net from scratch in NumPy
│
├── .env                           # Environment secrets (GEMINI_API_KEY) — gitignored
├── .gitignore                     # Git ignore rules protecting credentials, checkpoints, and .venv
└── README.md                      # Comprehensive project documentation
```

---

## 🌟 Gate H1: AI Landscape, LLMs & Tooling

### 1. AI Landscape Map (`AI_LANDSCAPE_MAP.md`)
An in-depth conceptual architecture and industry reference map across the five core subfields:
* **Machine Learning (ML):** Statistical boundary learning from tabular data (scikit-learn, XGBoost, LightGBM).
* **Deep Learning (DL):** Multi-layered artificial neural networks learning representations from unstructured data (PyTorch, TensorFlow, JAX, CUDA).
* **Natural Language Processing (NLP):** Linguistic parsing, tokenization, and semantic embeddings (Hugging Face Transformers, spaCy, vector databases).
* **Computer Vision (CV):** Visual perception, detection, and segmentation (OpenCV, torchvision, YOLO, SAM 2).
* **Generative AI (GenAI):** Foundation models synthesizing novel text, code, and multimodal assets (Google GenAI SDK, frontier LLMs, Diffusion).
* **12 Production Case Studies:** Deep dives into Stripe Radar, Netflix Recommendations, Zillow Zestimate, DeepMind AlphaFold, Tesla FSD, Google Translate, BloombergGPT, Apple Face ID, Waymo Driver, Gemini Code Assist, Midjourney, and Perplexity AI.

### 2. Structured LLM Response Pipeline (`structured_llm_response.ipynb`)
* **Secure Environment Credentials:** Loads `GEMINI_API_KEY` dynamically via `python-dotenv` with zero-leak key masking.
* **Modern SDK Integration:** Uses the official `google-genai` SDK (`genai.Client`) with `gemini-3.6-flash`.
* **Pydantic Data Contract:** Defines strict output schema with `AIConceptAnalysis(BaseModel)`.
* **Decode-Time Schema Enforcement:** Passes schema into `config={"response_mime_type": "application/json", "response_schema": AIConceptAnalysis}`.
* **Validation:** Parses and validates JSON via `AIConceptAnalysis.model_validate_json(response.text)`.

### 3. Toolchain & Environment Verification (`verify_env.py`)
* Python 3.12+ runtime confirmation.
* JupyterLab environment check.
* Hugging Face Hub connectivity check (queried `google/gemma-2-2b`).
* Kaggle CLI & tooling verification.
* End-to-end Gemini API call verification (`ENVIRONMENT_READY`).

---

## ⚡ Gate H2: Python for ML & First Pipeline

### 1. The Dataset: Credit Risk & Loan Default
* **Source:** Real-world credit risk dataset (32,581 records, 12 features).
* **Target:** `loan_status` (`0` = Non-default / Repaid, `1` = Default).
* **Data Nuances:** Mixed continuous numerical features, nominal categorical features, severe outliers, and missing values.

### 2. Data Cleaning & Outlier Purging (With Justifications)
* **Physically Impossible Outliers:** Purged rows where `person_age > 100` (e.g. age 144) and `person_emp_length > 60` or greater than `person_age - 14` (e.g. age 22 with 123 years of employment). Exactly **7 corrupt rows** removed.
* **Missing Value Justification:** Left missing entries in `loan_int_rate` (3,116 nulls) and `person_emp_length` (895 nulls) intact for imputation inside the pipeline rather than dropping them, preventing 10% data loss and training the model to handle incomplete forms.

### 3. Zero Data Leakage Architecture
* **Split Before Fitting:** `X_train` and `X_test` split (80/20) **before** fitting any scaler or imputer. Test statistics never contaminate the training phase.
* **Stratification:** Used `stratify=y` to guarantee identical default class ratios in both splits (`0.2182` train vs `0.2181` test).

### 4. Modular Preprocessing (`ColumnTransformer`)
* **Numerical Pipeline:** `SimpleImputer(strategy='median')` (robust to skewed incomes and loan amounts) + `StandardScaler()` (scales features to mean=0, variance=1).
* **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')` (converts categories to binary columns; safely ignores unseen categories in production).
* **Feature Expansion:** Transformed 11 raw features into **26 clean feature columns**.

### 5. Unified Model Pipeline & Evaluation
* **Pipeline Integration:** Assembled `Pipeline(steps=[('preprocessor', ...), ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12))])`.
* **Model Performance:**
  * **Overall Accuracy:** `92.97%`
  * **ROC-AUC Score:** `0.9220`
  * **Default (Class 1) Precision:** `0.98` (98% of flagged default risks are accurate; only 21 false positives out of 5,094 non-defaults).
  * **Default (Class 1) Recall:** `0.69` (Successfully intercepted 984 actual defaults).
  * **F1-Score:** `0.81` on defaults, `0.96` on non-defaults.

### 6. Production Inference & Assertions
* **Live Inference Test:** Tested with simulated new applicants, including one with missing values (`np.nan` in interest rate and employment length). The pipeline seamlessly imputed, encoded, scaled, and predicted without errors:
  * *Applicant 1 (Low Risk):* Approved (`0.60%` default risk).
  * *Applicant 2 (High Risk):* Rejected (`99.00%` default risk).
* **Automated Unit Tests / Assertions:**
  * Validated age $\le 100$ and employment $\le 60$.
  * Validated exactly zero nulls (`np.isnan`) and zero infinite values in transformed feature matrices.
  * Validated prediction outputs and valid class membership.

---

## 🔬 Gate H3: Math Behind ML

### 1. Pure Gradient Descent from Scratch (`math_behind_ml.ipynb` - Part 1)
* **Loss Function:** Mean Squared Error (MSE) on linear regression $y = w \cdot x + b$.
* **Analytical Gradients Derived:**
  $$\frac{\partial L}{\partial w} = \frac{2}{N} X^T (\hat{y} - y), \quad \frac{\partial L}{\partial b} = \frac{2}{N} \sum (\hat{y} - y)$$
* **Numerical Convergence:** Loss decreased monotonically from 65+ to near 0.
* **Benchmark Validation:** Matched Scikit-Learn's closed-form `LinearRegression` coefficients to 4 decimal places ($w = 2.8851, b = 5.1075$), passing tolerance assertions.

### 2. 2-Layer Neural Network with Analytical Backpropagation (Part 2)
* **Architecture:** Input ($N \times 2$) $\rightarrow$ Hidden ($N \times 4$, $\tanh$) $\rightarrow$ Output ($N \times 1$, $\text{Sigmoid}$).
* **Loss:** Binary Cross-Entropy (BCE).
* **Step-by-Step Chain Rule Derivations (Whiteboard Formulation):**
  1. Output Error: $dZ_2 = \hat{y} - y$
  2. Layer 2 Gradients: $dW_2 = \frac{1}{N} A_1^T dZ_2, \quad db_2 = \frac{1}{N} \sum dZ_2$
  3. Hidden Backpropagation: $dZ_1 = (dZ_2 W_2^T) \odot (1 - A_1^2)$ *(since $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$)*
  4. Layer 1 Gradients: $dW_1 = \frac{1}{N} X^T dZ_1, \quad db_1 = \frac{1}{N} \sum dZ_1$
  5. Gradient Step: $W \leftarrow W - \alpha \cdot dW, \quad b \leftarrow b - \alpha \cdot db$
* **No Frameworks:** 100% written in pure NumPy matrix vectorization with zero autograd.

### 3. Numerical Validation Against Scikit-Learn MLP (Part 3)
* **Dataset:** Non-linearly separable `make_moons` dataset.
* **Accuracy:** Reached **96.50% accuracy** from scratch (final BCE loss: `0.0824`).
* **Agreement Rate:** Achieved **100.00% prediction agreement** against `sklearn.neural_network.MLPClassifier`.
* **Visual Verification:** Plotted side-by-side decision boundaries showing identical non-linear separation curves.
* **Assertions:** Verified mathematical agreement and tolerance bounds programmatically.

---

## 🛠️ Setup & Execution Instructions

### 1. Environment Setup
Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

Install all dependencies:
```powershell
pip install python-dotenv pydantic google-genai jupyterlab ipykernel huggingface_hub kaggle pandas numpy scikit-learn matplotlib seaborn
```

### 2. Environment Variables
Ensure a `.env` file exists with your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Running the Notebooks
* **Gate H1 Notebook (Structured LLM Calls):**
  Open `structured_llm_response.ipynb`, select **`Python (.venv)`** kernel, and click **Run All**.
* **Gate H2 Notebook (ML Pipeline):**
  Open `ml_first_pipeline.ipynb`, select **`Python (.venv)`** kernel, and click **Run All**.
* **Gate H3 Notebook (Math Behind ML):**
  Open `math_behind_ml.ipynb`, select **`Python (.venv)`** kernel, and click **Run All**.
