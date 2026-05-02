import os
import re
import json
import requests
from typing import List, TypedDict, Literal

from pydantic import BaseModel, Field
import streamlit as st

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END

class OpenRouterLLM:
    def __init__(self, model="openai/gpt-4o-mini"):
        self.api_key = st.secrets["OPENROUTER_API_KEY"]
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def invoke(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

        response = requests.post(self.url, headers=headers, json=data)

        try:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print("FULL ERROR:", response.text)
            return f"Error: {response.text}"
            
llm = OpenRouterLLM()

file_path = "yt_data.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().replace("\n", " ")

docs = [Document(page_content=text)]
     
chunks = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=150
).split_documents(docs)
     

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
     

class State(TypedDict):
    question: str
    need_retrieval: bool

    docs: List[Document]
    relevant_docs: List[Document]

    context: str
    answer: str

    web_query: str
     

class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        ...,
        description="True if external data is required, else False."
    )

def build_decide_prompt(question: str) -> str:
    return f"""
You are a decision system in a RAG pipeline.

Decide whether answering the question requires external retrieval.

OUTPUT:
Return ONLY JSON:
{{"should_retrieve": true}} or {{"should_retrieve": false}}

RULES:

Return TRUE only if:
- Question depends on documents, PDFs, files
- Requires exact facts, citations, or up-to-date info
- Mentions "this document", "above", "file", etc.

Return FALSE if:
- General knowledge (e.g., "What is recursion?")
- Conceptual explanation
- Can be answered without external context

IMPORTANT:
- Default to FALSE
- Do NOT explain
- Do NOT hallucinate

Question:
{question}
"""

def decide_retrieval(state: State):
    prompt = build_decide_prompt(state["question"])
    response = llm.invoke(prompt)

    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            decision = RetrieveDecision(**data)
            return {"need_retrieval": decision.should_retrieve}
    except:
        pass

    return {"need_retrieval": False}
     

direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI learning assistant trained on Assamese educational content.\n\n"

            "LANGUAGE BEHAVIOR:\n"
            "- Match the user's language\n"
            "- If needed, translate knowledge from Assamese\n\n"

            "ANSWERING:\n"
            "- Use your general knowledge when no document context is available\n"
            "- Provide clear, student-friendly explanations\n\n"

            "STYLE:\n"
            "- Explain like a teacher (simple, structured, easy to understand)\n\n"

            "RULES:\n"
            "- Only answer study-related questions\n"
            "- If unrelated, say:\n"
            "  'I am an AI assistant designed to help students with their studies.'\n"
            "- Keep answers concise (2–4 lines)\n"
        ),
        ("human", "{question}"),
    ]
)

def generate_direct(state: State):
    messages = direct_generation_prompt.format_messages(
        question=state["question"]
    )

    prompt = "\n".join([m.content for m in messages])
    out = llm.invoke(prompt)

    return {
        **state,
        "answer": out
    }
     

def retrieve(state: State):
    return {
        **state,
        "docs": retriever.invoke(state["question"])
    }
     

class RelevanceDecision(BaseModel):
    is_relevant: bool

def build_relevance_prompt(question: str, document: str) -> str:
    return f"""
You are a strict relevance classifier.

Decide if the document helps answer the question.

OUTPUT:
Return ONLY JSON:
{{"is_relevant": true}} or {{"is_relevant": false}}

RULES:
- TRUE if document contains useful or related info
- FALSE if unrelated or weakly related
- Prefer FALSE if unsure

Question:
{question}

Document:
{document}
"""

def is_relevant(state: State):
    relevant_docs = []

    for doc in state["docs"]:
        prompt = build_relevance_prompt(
            state["question"],
            doc.page_content
        )

        response = llm.invoke(prompt)

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                decision = RelevanceDecision(**data)

                if decision.is_relevant:
                    relevant_docs.append(doc)
        except:
            pass

    return {
        **state,
        "relevant_docs": relevant_docs
    }
     

rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI learning assistant trained on Assamese educational content.\n\n"

            "LANGUAGE BEHAVIOR:\n"
            "- If the user asks in Assamese → answer in Assamese\n"
            "- If the user asks in another language → translate and answer using the Assamese content\n\n"

            "ANSWERING PRIORITY:\n"
            "- First, use the provided context (retrieved content)\n"
            "- If context is incomplete → use your general knowledge to complete the answer\n"
            "- Never say 'No relevant document found'\n\n"

            "STYLE:\n"
            "- Follow the same teaching style, tone, and simplicity as the original content\n"
            "- Keep explanations clear, simple, and student-friendly\n\n"

            "RULES:\n"
            "- Answer ONLY study-related questions\n"
            "- If unrelated, respond:\n"
            "  'I am an AI assistant designed to help students with their studies.'\n"
            "- Keep answers concise (2–4 lines unless needed)\n"
        ),
        (
            "human",
            "Answer this student question clearly:\n\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}"
        ),
    ]
)

def generate_from_context(state: State):
    context = "\n\n---\n\n".join(
        [d.page_content for d in state.get("relevant_docs", [])]
    ).strip()

    # fallback if no strong context
    if not context:
        messages = rag_generation_prompt.format_messages(
            question=state["question"],
            context="No strong context available"
        )

        prompt = "\n".join([m.content for m in messages])
        out = llm.invoke(prompt)

        return {
            **state,
            "answer": out,
            "context": ""
        }

    # normal RAG flow
    messages = rag_generation_prompt.format_messages(
        question=state["question"],
        context=context
    )

    prompt = "\n".join([m.content for m in messages])
    out = llm.invoke(prompt)

    return {
        **state,
        "answer": out,
        "context": context
    }
     
def route_after_decide(state: State):
    if state["need_retrieval"]:
        return "retrieve"
    return "generate_direct"
     

def route_after_relevance(state: State):
    return "generate_from_context"
     

g = StateGraph(State)

# Nodes
g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)
g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)

# Start
g.add_edge(START, "decide_retrieval")

# Decision routing
g.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {
        "generate_direct": "generate_direct",
        "retrieve": "retrieve",
    },
)

# Direct path ends
g.add_edge("generate_direct", END)

# Retrieval path
g.add_edge("retrieve", "is_relevant")

# Always go to generation (simplified for demo)
g.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
    },
)

# Final
g.add_edge("generate_from_context", END)

app = g.compile()
