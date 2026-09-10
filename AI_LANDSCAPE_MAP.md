# AI Landscape Map: Subfields, Tooling & Real-World Use Cases

Author: Misiko
Track: Track H - AI Foundations (Gate H1)

---

## 1. Executive Summary & Conceptual Hierarchy

Artificial Intelligence (AI) is the broad discipline of engineering systems capable of tasks that typically require human cognition. The relationships among the primary subfields can be viewed as an interconnected hierarchy:

- **Machine Learning (ML)**: A paradigm within AI where systems learn statistical associations and decision boundaries directly from data rather than relying on hand-coded deterministic rules.
- **Deep Learning (DL)**: A specialized subset of ML that uses deep artificial neural networks (ANNs) with multiple hidden layers. It learns hierarchical representations automatically without manual feature engineering.
- **Natural Language Processing (NLP)**: The domain focused on enabling computers to parse, interpret, extract, and manipulate human language.
- **Computer Vision (CV)**: The domain focused on acquiring, processing, analyzing, and understanding digital images and video to extract high-dimensional semantic features.
- **Generative AI (GenAI)**: The cutting-edge subfield built on deep foundation models (Transformers, Diffusion, Flow matching) that synthesize novel artifacts (text, code, photorealistic images, audio, molecules) rather than merely classifying or regressing observed data.

---

## 2. Deep Dive by Subfield

### Subfield 1: Machine Learning (ML)
- **Primary Paradigm**: Supervised learning (classification, regression), Unsupervised learning (clustering, dimensionality reduction), and Reinforcement Learning (policy/value optimization).
- **Dominant Data Type**: Tabular/structured data, time-series, relational database records.
- **Primary Tooling**:
  - scikit-learn (classical algorithms: Random Forests, SVM, Logistic Regression, k-means)
  - XGBoost / LightGBM / CatBoost (Gradient Boosted Decision Trees � state-of-the-art for tabular data)
  - pandas / 
umpy (data manipulation and vectorized numerical linear algebra)
  - MLflow / Weights & Biases (experiment tracking and model lifecycle)
- **Key Metric**: Precision, Recall, F1-Score, ROC-AUC, RMSE, MAE.

---

### Subfield 2: Deep Learning (DL)
- **Primary Paradigm**: Multi-layer perceptrons (MLP), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs/LSTMs), Graph Neural Networks (GNNs), and Transformers.
- **Dominant Data Type**: High-dimensional unstructured data (pixels, raw waveform audio, token sequences, graphs).
- **Primary Tooling**:
  - PyTorch (dominant research and production framework with dynamic computational graphs)
  - TensorFlow / Keras (production deployment, TFLite for edge devices)
  - JAX (high-performance composable function transformations, autodiff, and XLA compilation)
  - CUDA / cuDNN (NVIDIA hardware acceleration primitives)
  - TensorRT / ONNX Runtime (graph optimization and inference engines)
- **Key Metric**: Cross-entropy loss, Perplexity, Top-1/Top-5 accuracy.

---

### Subfield 3: Natural Language Processing (NLP)
- **Primary Paradigm**: Syntactic parsing, Named Entity Recognition (NER), Sentiment Analysis, Machine Translation, Information Retrieval, and Sequence-to-Sequence modeling.
- **Dominant Data Type**: Unstructured text corpora, documents, speech transcripts.
- **Primary Tooling**:
  - Hugging Face Transformers (pipeline, AutoModel, model hub)
  - spaCy (industrial-strength rule-based and neural NLP pipelines)
  - NLTK (foundational linguistic education and corpus processing)
  - 	iktoken / sentencepiece (subword tokenization algorithms: BPE, WordPiece, Unigram)
  - FAISS / ChromaDB / Qdrant (dense vector search for retrieval)
- **Key Metric**: BLEU, ROUGE, Exact Match (EM), cosine similarity.

---

### Subfield 4: Computer Vision (CV)
- **Primary Paradigm**: Image classification, Object detection (bounding boxes), Semantic & instance segmentation (pixel masks), Pose estimation, Optical Character Recognition (OCR).
- **Dominant Data Type**: 2D images, video streams, 3D point clouds, hyperspectral imaging.
- **Primary Tooling**:
  - OpenCV (real-time computer vision and image preprocessing)
  - 	orchvision (pre-trained CV backbones: ResNet, ViT, EfficientNet)
  - YOLO (Ultralytics YOLOv8/v11 for ultra-fast real-time object detection)
  - Albumentations (fast image augmentation pipelines)
  - Segment Anything (SAM / SAM 2) (zero-shot visual promptable segmentation)
- **Key Metric**: mAP (mean Average Precision @ IoU thresholds), IoU (Intersection over Union).

---

### Subfield 5: Generative AI (GenAI)
- **Primary Paradigm**: Autoregressive next-token prediction (Large Language Models), Denoising score matching (Diffusion models), Multimodal reasoning (vision-language-action models).
- **Dominant Data Type**: Multimodal tokens (interleaved text, images, video, code, tool calls).
- **Primary Tooling**:
  - google-genai SDK (Google Gemini API: Gemini 2.5/3 Flash & Pro)
  - OpenAI SDK / Anthropic SDK (frontier LLM interfaces)
  - LLM / Ollama / TGI (high-throughput serving with PagedAttention)
  - LangChain / LlamaIndex / CrewAI (agentic orchestration and RAG)
  - Diffusers (Hugging Face diffusion library for Stable Diffusion, Flux)
- **Key Metric**: HumanEval (code pass@1), MMLU, GPQA, Elo arena ratings, FID (Fr�chet Inception Distance for image quality).

---

## 3. Real-World Case Studies (12 Named Examples)

Here are 12 production-grade, named applications across the AI landscape:

| # | Subfield | Named System / Project | Organization | Technology Stack | Real-World Application & Mechanism |
|---|---|---|---|---|---|
| 1 | **ML** | **Stripe Radar** | Stripe | Gradient Boosted Trees, Custom ML pipelines | **Fraud Detection**: Evaluates billions of global payment transactions in <100ms. Scores risk based on historical card velocity, IP geolocation mismatch, and behavioral fingerprinting. |
| 2 | **ML** | **Netflix Recommendation Engine** | Netflix | Collaborative Filtering, Matrix Factorization, Contextual Bandits | **Personalized Content Discovery**: Dynamically ranks row carousels and customizes video thumbnail artwork per user profile based on viewing history, time of day, and retention curves. |
| 3 | **ML** | **Zillow Zestimate** | Zillow | XGBoost, Ensembled Regressors, Spatial ML | **Real Estate Valuation**: Estimates market value of over 100 million homes by combining property characteristics, regional tax records, macroeconomic indicators, and recent nearby sales comps. |
| 4 | **DL** | **AlphaFold 2 & 3** | Google DeepMind | Invariant Point Attention (IPA), Evoformer, Diffusion | **Computational Biology / Protein Structure**: Solved the 50-year-old protein folding grand challenge by predicting 3D atomic coordinates directly from 1D amino acid sequences. |
| 5 | **DL** | **Tesla Full Self-Driving (FSD v12+)** | Tesla | End-to-End Neural Networks, Occupancy Networks, PyTorch | **Autonomous Driving**: Replaced 300,000+ lines of explicit C++ heuristics with end-to-end deep neural networks mapping raw 360-degree photon camera streams directly into steering, braking, and throttle control. |
| 6 | **NLP** | **Google Translate** | Google | Transformer Seq2Seq, Multilingual NMT | **Real-Time Translation**: Translates across 100+ languages by mapping tokens into a shared semantic latent space, allowing direct zero-shot translation between low-resource language pairs. |
| 7 | **NLP** | **BloombergGPT** | Bloomberg LP | 50B Parameter Decoder Transformer, Domain-specific tokenization | **Financial NLP**: Trained on a 363-billion token financial dataset alongside general data to perform sentiment extraction from earnings calls, financial filings QA, and market headline classification. |
| 8 | **CV** | **Apple Face ID** | Apple | TrueDepth Sensor, Infrared Dot Projector, Neural Engine (CNNs) | **Biometric Security**: Projects 30,000 infrared dots onto the user's face to build a depth map. A neural network converts the mesh into a mathematical representation matching the stored biometric enrollment in <200ms. |
| 9 | **CV** | **Waymo Driver Perception** | Waymo (Alphabet) | 3D Convolutional Networks, PointNet, Multimodal Sensor Fusion | **Urban Obstacle Detection**: Fuses LiDAR point clouds, radar, and camera video to identify and track pedestrians, cyclists, construction cones, and emergency vehicles in dense urban environments. |
| 10 | **GenAI** | **Google Gemini Code Assist / Copilot** | Google / GitHub | Large Language Models (Gemini, Codex), Fill-in-the-Middle (FIM) | **AI Pair Programming**: Auto-completes syntax, generates unit tests, refactors functions, and synthesizes whole programs in IDEs by analyzing repository context and local imports. |
| 11 | **GenAI** | **Midjourney v6** | Midjourney | Latent Diffusion Models, Text-to-Image Cross-Attention | **Creative Asset Generation**: Generates high-fidelity, photorealistic or artistic images from natural language prompts by progressively denoising random Gaussian latent tensors conditioned on text embeddings. |
| 12 | **GenAI** | **Perplexity AI** | Perplexity | Retrieval-Augmented Generation (RAG), Sonar LLM, Search Indexes | **Conversational Search & Synthesis**: Replaces traditional keyword search engines by executing parallel web searches, fetching relevant documents, reading their contents, and generating verified, cited summaries. |

---

