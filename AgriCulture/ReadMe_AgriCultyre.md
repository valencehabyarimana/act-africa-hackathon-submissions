# Mkulima Agriculture AI Application Systems

## Project Files

```text
1_App_Agricultural_Knowledge_Activity.ipynb
2_App_AgriCultureAI_Agent.py
3_App_Agricultural_Knowledge_Activity.ipynb
Mkulima Knowledge_Base.xlsx
requirements.txt
ReadME_AgriCulture.md
```

---

## 1. Agricultural Knowledge Retrieval System

### Main File
`1_App_Agricultural_Knowledge_Activity.ipynb`

### Purpose
A practical retrieval system that searches the agricultural knowledge base using **TF-IDF and cosine similarity**.

### Libraries
```bash
pip install pandas scikit-learn openpyxl
```

### How to Run
Open the notebook in **Google Colab** or **Jupyter Notebook**.

Make sure:

```text
Mkulima Knowledge_Base.xlsx
```

is available, then run the notebook cells from top to bottom.

### Workflow
```text
Question
   ↓
TF-IDF Search
   ↓
Find Similar Question
   ↓
Retrieve Answer
```

---

## 2. Mkulima Agriculture AI Agent Assistant

### Main File
`2_App_AgriCultureAI_Agent.py`

### Purpose
A **Streamlit AI agriculture assistant** that retrieves relevant information from the Mkulima knowledge base and uses an LLM to generate the final answer.

### Libraries
Use the project requirements file:

```bash
pip install -r requirements.txt
```

Or install the main libraries:

```bash
pip install streamlit pandas sentence-transformers groq openpyxl torch
```

### How to Run

Keep the Python application and knowledge base in the same folder:

```text
2_App_AgriCultureAI_Agent.py
Mkulima Knowledge_Base.xlsx
```

Run:

```bash
streamlit run 2_App_AgriCultureAI_Agent.py
```

Enter the **Groq API key** in the application and ask an agriculture question.

### Models
- Embedding model: `all-MiniLM-L6-v2`
- LLM: `openai/gpt-oss-120b` through Groq

### Workflow
```text
Question
   ↓
Semantic Retrieval
   ↓
Mkulima Knowledge Base
   ↓
Groq LLM
   ↓
Final Answer
```

The application uses the knowledge base when the similarity score is at least **0.50**; otherwise, it uses general agriculture knowledge.

---

## 3. Mkulima AI Agricultural Knowledge Assistant

### Main File
`3_App_Agricultural_Knowledge_Activity.ipynb`

### Purpose
An advanced **RAG agricultural advisory system** that combines semantic retrieval, reranking, an open-source LLM, and a Gradio web interface with audio output.

### Libraries
The notebook uses libraries including:

```bash
pip install pandas scikit-learn openpyxl sentence-transformers transformers accelerate bitsandbytes torch gradio scipy numpy
```

### How to Run

Open the notebook in **Google Colab**.

1. Make `Mkulima Knowledge_Base.xlsx` available.
2. Run the notebook cells in order.
3. A GPU is recommended for the Qwen generation stage.
4. Run the final Gradio cell to launch the web application.
5. Use the interface to enter an agricultural question.

### Models
- Embedding model: `intfloat/multilingual-e5-base`
- Reranker: `BAAI/bge-reranker-v2-m3`
- LLM: `Qwen/Qwen2.5-3B-Instruct`
- Text-to-speech: `facebook/mms-tts-eng`

### Key Features
- TF-IDF retrieval
- Semantic retrieval
- E5 multilingual embeddings
- BGE reranking
- Qwen-based answer generation
- Gradio web interface
- Text and audio agricultural advice

### Workflow
```text
Question
   ↓
E5 Semantic Retrieval
   ↓
BGE Reranking
   ↓
Qwen 2.5 3B
   ↓
Final Agricultural Advice
   ↓
Text + Audio
```

---

## Supporting Files

### `Mkulima Knowledge_Base.xlsx`
Agricultural question-and-answer knowledge base used by the applications.

### `requirements.txt`
Contains the Python dependencies required for the project.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### `ReadME_AgriCulture.md`
Project documentation and usage information.

---

## Summary

| System | File | Main Technology | Run With |
|---|---|---|---|
| 1. Agricultural Knowledge Retrieval System | `1_App_Agricultural_Knowledge_Activity.ipynb` | TF-IDF + Cosine Similarity | Colab / Jupyter |
| 2. Mkulima Agriculture AI Agent Assistant | `2_App_AgriCultureAI_Agent.py` | Sentence Transformers + Groq + Streamlit | `streamlit run` |
| 3. Mkulima AI Agricultural Knowledge Assistant | `3_App_Agricultural_Knowledge_Activity.ipynb` | E5 + BGE + Qwen + Gradio + TTS | Google Colab |
