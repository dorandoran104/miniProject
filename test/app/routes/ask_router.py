from fastapi import APIRouter, File, UploadFile, Form

from service import ask_service

import time

ask = APIRouter()

@ask.post("/question")
async def ask_route(file: UploadFile = File(...), question: str = Form(...)):

    read_time = time.time()
    docs = await ask_service.read_pdf(file)
    read_end_time = time.time()
    print(f"Read time: {read_end_time - read_time}")

    embedding_time = time.time()
    embedding = await ask_service.create_embedding()
    embedding_end_time = time.time()
    print(f"Embedding time: {embedding_end_time - embedding_time}")
    
    #검색기 생성
    create_vector_store_time = time.time()
    vector_store = await ask_service.create_vector_store(docs, embedding)
    create_vector_store_end_time = time.time()
    print(f"Create vector store time: {create_vector_store_end_time - create_vector_store_time}")
    # retriever = vector_store.as_retriever()

    #검색
    ask_time = time.time()
    result = await ask_service.search_vector_store(question,vector_store)
    ask_end_time = time.time()
    print(f"Ask question time: {ask_end_time - ask_time}")
    print(result)
    
    return {"message": result}

 