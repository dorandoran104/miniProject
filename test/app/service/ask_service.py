from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import UploadFile
from langchain.vectorstores import Chroma

import tempfile
import dotenv
import os
import google.generativeai as genai
import logging

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
generation_config = {
  "temperature": 0,
  "top_p": 0.6,
  "top_k": 2,
  "max_output_tokens": 500,
}

#pdf 파일 읽기
async def read_pdf(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_content = await file.read()
        temp_file.write(file_content)
        temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = []
        # pages.extend(loader.load())
        async for page in loader.alazy_load():
            pages.append(page)
        return pages
        return pages

async def split_pdf(docs):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = text_spliter.split_documents(docs)
    return docs
    
#임베딩 생성
async def create_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings
    # vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    # return vector_store.similarity_search(question,k=3)

#벡터 저장소 생성
async def create_vector_store(docs, embeddings,file_name):
    # vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=f"vector_store/{file_name}",
        collection_name=file_name
    )
    vector_store.persist()
    return vector_store

#벡터 저장소 로드
async def load_vector_store(file_name,embeddings):
    vector_store = Chroma(
        persist_directory=f"vector_store/{file_name}",
        collection_name=file_name,
        embedding_function=embeddings
        )
    
    return vector_store

#벡터 저장소 검색
async def search_vector_store(question,vector_store,embeddings): 
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-001",config=generation_config)
    retriever = vector_store.as_retriever(
        embedding_function=embeddings
        ,search_kwargs={"k": 2}
        ,search_type="mmr"
        )
    retriever_from_llm = MultiQueryRetriever.from_llm(llm=llm,retriever=retriever)

    prompt = PromptTemplate.from_template(
        """
        PDF 문서에서 주어진 질문에 대한 답변을 제공하는 프롬프트입니다.
        주어진 문서는 텍스트와 표로 구성되어 있습니다.
        표 항목도 있으므로 표 항목은 주변 텍스트를 확인하고 답변해주세요.
        숫자로 이루어진 표는 되도록 합계쪽을 읽고 대답해주세요
        주어진 문서를 참고하여 질문에 대한 자세한 답변을 제공하고, 관련된 경우 목차를 포함하여 추가적인 지침을 제공합니다.
        답변은 너무 길지 않게 요약해서 한국어로만 대답해주세요
        
        #Contents:
        {context}

        #Question:
        {question}

        #Answer:"""
    )

    # prompt = PromptTemplate.from_template(
    #     """
    #     PDF 문서에서 주어진 질문에 대한 답변을 제공하는 프롬프트입니다.
    #     주어진 문서를 참고하여 질문에 대한 자세한 답변을 제공하고, 관련된 경우 목차를 포함하여 추가적인 지침을 제공합니다.
        
    #     표 항목도 있으므로 표 항목은 주변 텍스트를 확인하고 답변해주세요.
    #     답변은 한국어로만 제공됩니다.
    #     너무 길지 않게 요약해서 답변해주세요

    #     #Context: 
    #     {context}

    #     #Question:
    #     {question}

    #     #Answer:"""
    #     )
    chain = (
        {
            "context": retriever_from_llm,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(question)
    return result
