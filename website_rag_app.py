import time
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
ENV = r"E:\gen_ai_projects\shared_keys\.env"
load_dotenv(ENV) 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file. Please add it.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add it.")

print("✅ Environment variables loaded successfully!")

urls =['https://www.victoriaonmove.com.au/local-removalists.html','https://victoriaonmove.com.au/index.html','https://victoriaonmove.com.au/contact.html']
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

#print(len(docs))
#print(docs[0])


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorestore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorestore.as_retriever(search_type="similarity",search_kwargs={"k":6})
"""
retrieved_docs =retriever.invoke("What kind of services they provide?")
for i, doc in enumerate(retrieved_docs):
    print("\n--- DOC", i, "---")
    print(doc.page_content)
"""
#llm 
llm = ChatGroq(model_name = "llama-3.1-8b-instant",
                temperature = 0.1,
                max_tokens = 500,
                )
system_template = """
You are an advanced AI RAG Assistant.

Your job is to answer the user's question ONLY from the provided website context.

========================
WEBSITE CONTEXT:
{context}
========================

USER INPUT:
{input}

INSTRUCTIONS:
- Use only the provided context.
- Do not hallucinate or invent information.
- If the answer is not available, say:
  "I could not find this information in the website data."
- Give concise, accurate, and structured answers.
- Combine relevant information from multiple chunks if needed.
- Use bullet points when appropriate.
- If possible, mention the relevant webpage or section.
- Ignore irrelevant retrieved text.
- Maintain a professional and intelligent tone.

OUTPUT FORMAT:
1. Direct Answer
2. Key Details
3. Additional Context (optional)

ANSWER:
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_template),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "What kind of services do they provided?"})
print(response["answer"]) 


