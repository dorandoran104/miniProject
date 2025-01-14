from fastapi import APIRouter, File, UploadFile, Form

from service import ask_service
from langchain_teddynote import logging

import time
import os

ask = APIRouter()

logging.langsmith("LAG_TEST")

@ask.post("/question")
async def ask_route(file: UploadFile = File(...), question: str = Form(...)):

    file_name = file.filename

    if not os.path.exists(f"vector_store/{file_name}"):

        read_time = time.time()
        docs = await ask_service.read_pdf(file)
        read_end_time = time.time()
        print(f"Read time: {read_end_time - read_time}")

        embedding_time = time.time()
        embedding = await ask_service.create_embedding()
        embedding_end_time = time.time()
        print(f"Embedding time: {embedding_end_time - embedding_time}")

        # 텍스트 분리
        split_time = time.time()
        split_docs = await ask_service.split_pdf(docs)
        split_end_time = time.time()
        print(f"Split time: {split_end_time - split_time}")
        
        #검색기 생성
        create_vector_store_time = time.time()
        vector_store = await ask_service.create_vector_store(split_docs, embedding,file_name)
        create_vector_store_end_time = time.time()
        print(f"Create vector store time: {create_vector_store_end_time - create_vector_store_time}")
    else:
        embedding_time = time.time()
        embedding = await ask_service.create_embedding()
        embedding_end_time = time.time()
        print(f"Embedding time: {embedding_end_time - embedding_time}")

        vector_store_load_time = time.time()
        vector_store = await ask_service.load_vector_store(file_name,embedding)   
        vector_store_load_end_time = time.time()
        print(f"Vector store load time: {vector_store_load_end_time - vector_store_load_time}")

    #검색
    ask_time = time.time()
    result = await ask_service.search_vector_store(question,vector_store,embedding)
    ask_end_time = time.time()
    print(f"Ask question time: {ask_end_time - ask_time}")
    print(result)
    
    return {"message": result}

 