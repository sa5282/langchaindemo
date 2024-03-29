from customllm import CustomLLM
from customllm import default_llm
from custom_embedders import CustomChromaEmbedder
from custom_embedders import default_embedder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

question = "What is the CHIPS Act?"
template = """Instructions: Use only the following context to answer the question.

Context: {context}
Question: {question}
"""

# embedder = CustomChromaEmbedder()
embedder = default_embedder
db = Chroma(persist_directory="./chromadb", embedding_function=embedder)
retriever = db.as_retriever(search_kwargs={"k": 1})
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# llm = CustomLLM()
llm = default_llm
# chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke(question))
print("#----------------------#")


recipe_prompt = PromptTemplate.from_template(
    """You are a chef. Given the title of a dish, it is your job to write a recipe for that dish.

Name: {name}
Chef: This is a recipe for the above dish:"""
)

review_prompt = PromptTemplate.from_template(
    """You are a food critic for a magazine. Given the recipe of a dish, it is your job to write a review for that dish.

Dish Synopsis:
Review from a food critic of the above recipe:"""
)

# chain = (
#     recipe_prompt
#     |llm
#     |StrOutputParser
#     #{"recipe": recipe_prompt | llm | StrOutputParser()}
#     #| review_prompt
#     #| llm
#     #| StrOutputParser()
# )
# print(chain.invoke({"name": "shrimp Gumbo"}))