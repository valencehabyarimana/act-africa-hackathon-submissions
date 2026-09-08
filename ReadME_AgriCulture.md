# Mkulima Agriculture AI Application Systems

This document describes the application systems for the Mkulima Agriculture AI project.

## 1. Agricultural Knowledge Retrieval System

### Purpose
Retrieves the most relevant agricultural question and answer from the knowledge base using TF-IDF and cosine similarity.

### Main File
`Agricultural_Knowledge_Activity.ipynb`

### Libraries to Install
```bash
pip install pandas scikit-learn openpyxl
```

### How to Run
**Google Colab**
1. Open `Agricultural_Knowledge_Activity.ipynb` in Google Colab.
2. Make `Mkulima Knowledge_Base.xlsx` available.
3. Run the cells from top to bottom.
4. Enter/change the agricultural question in the search section.

**Local Jupyter**
```bash
jupyter notebook Agricultural_Knowledge_Activity.ipynb
```

### Workflow
`Question → TF-IDF Search → Similar Question → Retrieved Answer`

---

## 2. Mkulima Agriculture AI Agent Assistant

### Purpose
An interactive Streamlit application that retrieves relevant agricultural knowledge and uses an LLM to generate the final answer.

### Main File
`app_.py`

### Libraries to Install
```bash
pip install streamlit pandas sentence-transformers groq openpyxl torch
```

### How to Run
Keep these files in the same folder:

```text
app_.py
Mkulima Knowledge_Base.xlsx
```

Run:

```bash
streamlit run app_.py
```

Then:
1. Enter the Groq API key.
2. Enter an agricultural question.
3. The system retrieves relevant knowledge.
4. The LLM generates the final response.

### Models
- Embedding model: `all-MiniLM-L6-v2`
- LLM: `openai/gpt-oss-120b` through Groq

### Workflow
`Question → Semantic Retrieval → Knowledge Base → LLM → Final Answer`

---

## 3. Application System

**Reserved for the third application system.**

### Purpose
_To be added._

### Main File
_To be added._

### Libraries to Install
_To be added._

### How to Run
_To be added._

### Models
_To be added._

### Workflow
_To be added._

---

## Summary

| System | Main Technology | How to Run |
|---|---|---|
| 1. Agricultural Knowledge Retrieval System | TF-IDF + Cosine Similarity | Google Colab / Jupyter |
| 2. Mkulima Agriculture AI Agent Assistant | Sentence Transformers + Groq LLM + Streamlit | `streamlit run app_.py` |
| 3. Application System | Reserved | To be added |
