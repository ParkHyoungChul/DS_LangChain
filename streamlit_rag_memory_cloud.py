import os
import streamlit as st

from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

from langchain_chroma import Chroma

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=persist_directory
    )
    return vectorstore

@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
        )
    else:
        return create_vector_store(_docs)

@st.cache_resource
def initialize_components(selected_model):
    file_path = r"./대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt),
         MessagesPlaceholder("history"),
         ("human", "{input}")]
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer perfect. please use imogi with the answer.\n"
        "대답은 한국어로 하고, 존댓말을 써줘.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt),
         MessagesPlaceholder("history"),
         ("human", "{input}")]
    )

    llm = ChatOpenAI(model=selected_model)
    rewrite_chain = contextualize_q_prompt | llm | StrOutputParser()

    history_aware_retriever = (
        {"input": itemgetter("input"), "history": itemgetter("history")}
        | rewrite_chain
        | retriever
    )

    def format_docs(docs):
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    # ✅ dict로 반환: {"answer": ..., "context_docs": [...]}
    rag_chain = {
        "context_docs": history_aware_retriever,
        "answer": (
            {
                "context": history_aware_retriever | RunnableLambda(format_docs),
                "input": itemgetter("input"),
                "history": itemgetter("history"),
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        ),
    }
    return rag_chain

# UI
st.header("헌법 Q&A 챗봇 💬 📚")
option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))
rag_chain = initialize_components(option)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",  # ✅ dict에서 어떤 키를 'assistant 발화'로 기록할지 지정
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "헌법에 대해 무엇이든 물어보세요!"}]

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_message := st.chat_input("Your question"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)

            # ✅ dict 구조에 맞게 출력
            st.write(response["answer"])
            with st.expander("참고 문서 확인"):
                for doc in response["context_docs"]:
                    # source가 없을 수도 있으니 .get 사용 추천
                    src = doc.metadata.get("source", "(source 없음)")
                    st.markdown(src, help=doc.page_content)
