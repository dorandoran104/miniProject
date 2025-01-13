from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains import AnalyzeDocumentChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

from fastapi import UploadFile
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
        async for page in loader.alazy_load():
            pages.append(page)
        return pages
    
#임베딩 생성
async def create_embedding():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings
    # vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    # return vector_store.similarity_search(question,k=3)

#벡터 저장소 생성
async def create_vector_store(docs, embeddings):
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    return vector_store

#벡터 저장소 검색
async def search_vector_store(question,vector_store): 
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-001",config=generation_config)
    
    retriever_from_llm = MultiQueryRetriever.from_llm(llm=llm,retriever=vector_store.as_retriever())
   

    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    unique_docs = retriever_from_llm.get_relevant_documents(query=question)

    print(len(unique_docs))
    print()

    for i in unique_docs:
        print('-'*20)
        print(i)
        print('-'*20)

    prompt = PromptTemplate.from_template(
        """
        PDF 문서에서 주어진 질문에 대한 답변을 제공하는 프롬프트입니다.
        주어진 문서를 참고하여 질문에 대한 자세한 답변을 제공하고, 관련된 경우 목차를 포함하여 추가적인 지침을 제공합니다.
        답변은 한국어로만 제공됩니다.
        숫자로 이뤄진 표 항목도 있으므로 표 항목은 형태를 확인하고 답변해주세요.

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
        )
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
    print(chain.invoke(question))
    return result
