from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# import dotenv
# dotenv.load_dotenv()

import sqlite_fix

# loading from disk
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./book_db", embedding_function=embedding)

# similarity search capabilities of a vector store to facillitate retrieval
retriever = vectorstore.as_retriever(**{"n_results": 2})

print(retriever)

# prompt = hub.pull("rlm/rag-prompt")

chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Artificial Intelligence instructor.
               You have knowledge of topics regarding Artificial Intelligence.""",
        ),
        (
            "human",
            """You are an assistant for question-answering tasks.
              Use the following pieces of retrieved context to answer the question.
              If you don't know the answer, just say that you don't know.
              Use five sentences maximum and keep the answer concise.
               Question: {question}
               Context: {context}
               Answer:""",
        ),
    ]
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

import json

def doc_display(doc):
    return doc.page_content + '\n***' + json.dumps(doc.metadata)

def format_docs(docs):
    # print("\n\n".join(doc.page_content for doc in docs))
    # return "\n========\n\n".join(doc_display(doc) for doc in docs) 
    return "\n\n".join(doc.page_content for doc in docs)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


# LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # RunnablePassthrough.assign(
    #     context=contextualized_question | retriever | format_docs
    # )
    | chat_template
    | llm
    | StrOutputParser()
)

chat_history = []

def main():
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        result1 = rag_chain.invoke(user_input)
        print(result1)
        chat_history.extend(
            [HumanMessage(content=user_input), AIMessage(content=result1)]
        )


if __name__ == "__main__":
    main()
