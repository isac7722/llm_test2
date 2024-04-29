# API 키를 환경 변수로 관리하기 위한 설정 파일
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ChatMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from utils import print_messages, StreamHandler
import streamlit as st
from langchain_core.prompts import PromptTemplate
from main import ensemble_retriever

from dotenv import load_dotenv
# API 키 정보 로드
load_dotenv()


st.set_page_config(page_title="LangChain Master", page_icon="👨🏻‍💻")
st.title("LangChain Master 👨🏻‍💻")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 채팅 대화기록을 저장하는 store 세션 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

with st.sidebar:
    session_id = st.text_input("Session ID", value="root")

    clear_btn = st.button("대화기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["store"] = dict()
        st.rerun()

# 이전 대화기록을 출력해주는 코드
print_messages()


def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


if user_input := st.chat_input("메세지를 입력해주세요"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(
        ChatMessage(role="user", content=user_input))

    # LLM을 사용하여 AI의 답변을 생성

    # AI의 답변
    with st.chat_message("assistant"):
        # stream_handler = StreamHandler(st.empty())

        # # 1. 모델 생성
        # llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])




        ## LLM 정의
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain_core.callbacks.manager import CallbackManager
        from langchain_core.runnables import ConfigurableField
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        from langchain_community.chat_models import ChatOllama
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic

        class StreamCallback(BaseCallbackHandler):
            def on_llm_new_token(self, token:str, **kwargs):
                print(token, end="", flush=True)

        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            streaming=True,
            callbacks=[StreamCallback()],
        ).configurable_alternatives(
            # 이 필드에 id를 부여합니다
            # 최종 실행 가능한 객체를 구성할 때, 이 id를 사용하여 이 필드를 구성할 수 있습니다.
            ConfigurableField(id="llm"),
            # 기본 키를 설정합니다.
            default_key="gpt4",
            claude=ChatAnthropic(
                model="claude-3-opus-20240229",
                temperature=0,
                streaming=True,
                callbacks=[StreamCallback()],
            ),
            gpt3=ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                streaming=True,
                callbacks=[StreamCallback()],
            ),
            ollama=ChatOllama(
                model="EEVE-Korean-10.8B:long",
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            ),
        )





        # 2. 프롬프트 생성
        
        prompt = PromptTemplate.from_template(
            """
                당신은 20년차 AI 개발자입니다. 당신의 임무는 주언지 질문에 대하여 최대한 문서의 정보를 활용하여 답변하는 것입니다.
                문서는 Python 코드에 대한 정보를 담고 있습니다. 따라서, 답변을 작성할 때에는 Python 코드에 대한 상세한 code snippet을 포함하여 작성해주세요.
                최대한 자세하게 답변하고, 한글로 답변해 주세요. 주어진 문서에서 답변을 찾을 수 없는 경우, "문서에 답변이 없습니다."라고 답변해 주세요.
                답변은 출처(source)를 반드시 표기해 주세요.

                #참고문서:
                {context}

                #질문:
                {question}

                #답변:

                출처:
                - source1
                - source2
                - ...
            """
        )

        # chain = prompt | llm


        rag_chain = (
            {"context": ensemble_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )







        # chain_with_memory = (
        #     RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
        #         rag_chain,  # 실행할 Runnable 객체
        #         get_session_history,  # 세션 기록을 가져오는 함수
        #         input_messages_key="question",  # 사용자 질문의 키
        #         history_messages_key="history",  # 기록 메세지의 키
        #     )
        # )

        # response = chain.invoke({"question":user_input})
        response = rag_chain.invoke(

            # {"question": user_input},
            {user_input},
            # # session ID 설정
            # config={"configurable": {"session_id": session_id}},
        )

        msg = response.content

        # st.write(msg)
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=msg))
