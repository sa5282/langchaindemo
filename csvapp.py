from langchain_community.document_loaders.csv_loader import CSVLoader

from custom_embedders import default_embedder
from langchain_community.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from provider_data import ProviderData
from customllm import default_llm
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


loader = CSVLoader(file_path="./documents/input-2.csv", encoding="utf-8", csv_args={'delimiter': ';'})
data = loader.load()

llm = default_llm
embedder = default_embedder
db = Chroma.from_documents(data, embedding=embedder)

query = "Provide the information on Provider identified by ID IKM05 in the json format"
#docs = db.similarity_search(query, 1)
retriever = db.as_retriever(search_kwargs={"k": 1})

print(retriever)

#parser = PydanticOutputParser(pydantic_object=ProviderData)
parser = JsonOutputParser(pydantic_object=ProviderData)

# prompt = PromptTemplate(
#     template="Answer the user query.\n{format_instructions}\n{query}\n",
#     input_variables=["query"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

prompt = PromptTemplate(
    template="Answer the user query as best as possible using the provided context.\n{context}\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

#print(prompt)

chain = ( 
            {"context": retriever, "query": RunnablePassthrough()}
            | prompt
            | llm 
            | parser
        )

print(chain.invoke(query))