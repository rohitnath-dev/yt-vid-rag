import streamlit as st
import time

# import your RAG pipeline
from rag import app as rag_app

st.set_page_config(page_title="AI Assistant")

st.title("You Can Learn AI Assistant")
st.write(
    "You can ask anything from videos, notes, contents or your class studies."
)

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- INPUT ----------
user_input = st.chat_input("Ask your doubt...")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # generate response
    result = rag_app.invoke({
        "question": user_input,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
    })

    answer = result["answer"]

    # typing animation
    with st.chat_message("assistant"):
        placeholder = st.empty()
        typed = ""

        for char in answer:
            typed += char
            placeholder.markdown(typed)
            time.sleep(0.01)

    # save AI message
    st.session_state.messages.append({"role": "assistant", "content": answer})
