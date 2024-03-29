from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader

from custom_embedders import default_embedder
from langchain_community.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from provider_data import ProviderData
from customllm import default_llm
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


loader = CSVLoader(file_path="./documents/input-2.csv", encoding="utf-8", csv_args={'delimiter': ';'})
data = loader.load()

print(f"data size = {len(data)}")

loader = CSVLoader(file_path="./documents/input-3.csv", encoding="utf-8", csv_args={'delimiter': ';'})
data2 = loader.load()

print(f"data2 size = {len(data2)}")


question = "What is the first and last of the Provider identified by IKM05 and is that provider accepting new patients? My insurance plan is Plan-5, do you know if that provider accepts this insurance plan?"
#question = "Do you know any Allergist? If you do then please provide me the all the details you have for that provider."
question = "I need to know the all the details about the provider Larry Marshall including his office hours for every day of the week"


template = """Instructions: Use only the following context to answer the question.

Context: {context}
Question: {question}
"""

llm = default_llm
embedder = default_embedder
db = Chroma.from_documents(data, embedding=embedder)
db = db.from_documents(data2, embedding=embedder)

retriever = db.as_retriever(search_kwargs={"k": 2})

print(retriever)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke(question))