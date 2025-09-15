import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# Load env variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

app = FastAPI(title="Warehouse AI Support")

# Request model
class QuestionRequest(BaseModel):
    question: str

# Initialize RAG pipeline once
try:
    print("🔄 Connecting to Qdrant...")

    # 1️⃣ Embeddings model (same as in embedding.py)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2️⃣ Connect to Qdrant Cloud
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    # 3️⃣ Load existing collection as retriever
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name="warehouse_docs",
        embeddings=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4️⃣ Warehouse-focused prompt
    prompt_template = """
You are a warehouse AI assistant.

Your job:
- Use the provided context first.  
- If context is insufficient, rely on your own warehouse knowledge.  
- Always answer like a seasoned warehouse operations manager: direct, sharp, confident.  

Answering Rules:
- Keep it concise (2–4 sentences max).  
- Use only warehouse/operations terms.  
- Provide small, simple real-world examples (1–3) if relevant.  
- Avoid theory, avoid long explanations.  
- If truly unknown, reply only: "I don't know."  
- If you know, explain in short and sweet with some decoration.  

Context:
{context}

Question: {question}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 5️⃣ LLM
    llm = ChatOpenAI(
        model="mistralai/mistral-small-3.2-24b-instruct:free",  # or any OpenRouter-supported model
        temperature=0.1,
        openai_api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    # 6️⃣ RetrievalQA pipeline
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    print("✅ FastAPI connected to Qdrant successfully!")

except Exception as e:
    print("❌ Error initializing RAG pipeline:", e)
    qa = None


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if qa is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not available.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = qa.invoke({"query": question})
        if response is None:
            raise Exception("No response from the model")

        response_text = response.get("result", "") if isinstance(response, dict) else str(response)

        if not response_text:
            raise Exception("Empty response from the model")

        return {
            "question": question,
            "answer": response_text.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
