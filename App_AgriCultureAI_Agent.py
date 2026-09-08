import os
import pandas as pd
import streamlit as st

from groq import Groq
from sentence_transformers import SentenceTransformer, util


# --------------------------------------------------
# 1. Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Mkulima Agriculture AI Agent Assistant",
    page_icon="🌱"
)

st.subheader("🌱 Mkulima Agriculture AI Agent Expert Assistant")


# --------------------------------------------------
# 2. Groq API Key
# --------------------------------------------------

api_key = st.text_input(
    "Enter your Groq API Key",
    type="password"
)

if not api_key:
    st.info("Please enter your Groq API key.")
    st.stop()

client = Groq(api_key=api_key)


# --------------------------------------------------
# 3. Load knowledge base
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_excel(
        "Mkulima Knowledge_Base.xlsx"
    )

    return df[["Qestions", "Answers"]].dropna()


df = load_data()


# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedder = load_embedding_model()

embeddings = embedder.encode(
    df["Qestions"].tolist(),
    convert_to_tensor=True
)


# --------------------------------------------------
# 5. Ask question
# --------------------------------------------------

question = st.text_input(
    "Ask an agriculture question:"
)


if st.button("Ask") and question:

    # Find similar question
    question_embedding = embedder.encode(
        question,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        question_embedding,
        embeddings
    )[0]

    best_index = scores.argmax().item()
    similarity = scores[best_index].item()

    retrieved_question = df.iloc[best_index]["Qestions"]
    retrieved_answer = df.iloc[best_index]["Answers"]


    # --------------------------------------------------
    # 6. RAG or general knowledge
    # --------------------------------------------------

    if similarity >= 0.50:

        context = f"""
Knowledge Base Question:
{retrieved_question}

Knowledge Base Answer:
{retrieved_answer}
"""

        source = "Mkulima Knowledge Base"

    else:

        context = """
No sufficiently similar information was found
in the Mkulima knowledge base.

Use your general agriculture knowledge.
"""

        source = "General Agriculture Knowledge"


    # --------------------------------------------------
    # 7. Ask Groq
    # --------------------------------------------------

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an agriculture expert. "
                    "Give clear and practical answers. "
                    "Answer in the same language as the question."
                )
            },

            {
                "role": "user",
                "content": f"""
Question:
{question}

Relevant information:
{context}

Answer the question.
"""
            }
        ],

        temperature=0
    )


    answer = response.choices[0].message.content


    # --------------------------------------------------
    # 8. Display answer
    # --------------------------------------------------

    st.subheader("Answer")

    st.write(answer)

    st.caption(f"Source: {source}")


    # --------------------------------------------------
    # 9. Show retrieved information
    # --------------------------------------------------

    with st.expander("Retrieved Knowledge"):

        st.write(
            f"Similarity score: {similarity:.3f}"
        )

        if similarity >= 0.50:

            st.write("**Matched Question:**")
            st.write(retrieved_question)

            st.write("**Knowledge Base Answer:**")
            st.write(retrieved_answer)

        else:

            st.write(
                "No sufficiently similar information "
                "was found in the knowledge base."
            )