from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

# Load documents
loader = TextLoader("data/admissions_faq.txt")
documents = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)

# Use HF model (Mistral, Falcon, etc.)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here"
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature":0.5, "max_new_tokens":200})

# Retrieval QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Chat loop
print("🎓 College Admission Agent Ready!")
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    response = qa.run(query)
    print("🧠 Answer:", response)
