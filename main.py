import os
from dotenv import load_dotenv

load_dotenv()

# langsmith 설정
# os.environ['LANGCHAIN_TRACING_V2'] = "true"
# os.environ['LANGCHAIN_PROJECT'] = "Code Analysis"

# Root 경로
repo_root = "/Users/user/Documents/PangJoong/langchain/libs"

# 불러오고자 하는 패키지 경로
repo_core = repo_root + "/core/langchain_core"
repo_community = repo_root + "/community/langchain_community"
repo_experimental = repo_root + "/experimental/langchain_experimental"
repo_partners = repo_root + "/partners"
repo_text_splitter = repo_root + "/text_splitters/langchain_text_splitters"
repo_cookbook = repo_root + "/cookbook"

# langchain 의 여러 모듈을 가져온다
from langchain_text_splitters import Language
from langchain.document_loaders.generic import  GenericLoader
from langchain.document_loaders.parsers import LanguageParser

# 불러온 문서를 저장할 빈 리스트를 생성
py_documents = []

for path in [repo_core, repo_community, repo_experimental, repo_partners, repo_cookbook]:
    # GenericLoader를 사용하여 파일 시스템에서 문서를 로드합니다.
    loader = GenericLoader.from_filesystem(
        path, # 문서를 불러올 경로
        glob="**/*",
        suffixes=[".py"], # py 확장자를 가진 파일만 대상으로 함
        parser=LanguageParser(
            language=Language.PYTHON, parser_threshold=30
        ), # 파이썬 언어의 문서를 파싱하기 위한 설정
    )
    # 로더를 통해 불러온 문서들을 documents 리스트에 추가합니다.
    py_documents.extend(loader.load())

print(f".py 파일의 계수: {len(py_documents)}")


# TextLoader 모듈을 불러온다
from langchain_community.document_loaders import TextLoader

# 검색할 최상위 디렉토리 경로를 정의합니다
root_dir = "/Users/user/Documents/PangJoong/langchain/"

mdx_documents = []
# os.walk를 사용하여 root_dir부터 시작하는 모든 디렉토를 순회한다
for dirpath, dirnames, filenames in  os.walk(root_dir):
    # 각 디렉토리에서 파일 목록을 확인한다
    for file in filenames:
        # TextLoader를 사용하여 파일의 전체 경로를 지어하고 문서를 로드합니다.
        if(file.endswith(".mdx")) and "*venv/" not in dirpath:
            try:
                # TextLoader를 사용하여 파일의 전체 경로를 지정하고 문서를 로드합니다.
                loader = TextLoader(os.path.join(
                    dirpath,file), encoding="utf-8")
                # 로드한 문서를 분할하여 documents 리스트에 추가합니다
                mdx_documents.extend(loader.load())
            except Exception:
                # 파일 로드 중 오ㄹㅍ가 발생하면 이를 무시하고 계속 진행합니다.
                pass

# 최종적으로 불러온 문서의 개수를 출력한다
print(f".mdx 파일의 개수: {len(mdx_documents)}")


# RecursiveCharacterTextSplitter 모듈을 가져온다
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RecursiveCharacterTextSplitter 객체를 생성. 이때, 파이썬 코드를 대상으로 하며,
# 청크 크기는 2000, 청크간 겹치는 부분은 200 문자로 설정한다
py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)

# py_docs 변수에 저장된 문서들을 위에서 설정한 청크 크기와 겹치는 부분을 고려하여 분할한다
py_docs = py_splitter.split_documents(py_documents)

# 분할된 텍스트의 개수를 출력한다
print(f"분할된 .py파일의 개수 : {len(py_docs)}")

mdx_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# mdx_docs 변수에 저장된 문서들을 위에서 설정한 청크 크기와 겹치는 부분을 고려하여 분할한다
mdx_docs = mdx_splitter.split_documents(mdx_documents)

# 분할된 텍스트의 개수를 출력한다
print(f"분할된 .mdx 파일의 개수: {len(mdx_docs)}")


combined_documents = py_docs + mdx_docs
print(f"총 문서 개수 : {len(combined_documents)}")

 


 ########## EMBEDDINGS################
# langchain_openai 와 langchain의 필요한 모듈들을 가져옵니다
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# 로컬 파일 저장소를 사용하기 위해 LocalFileStore 인스턴스를 생성한다
# './cache/' 디렉토리에 데이터를 저장합니다.
store = LocalFileStore("./cache/")

# OpenAI 임베딩 모델 인스턴스를 생성합니다. 모델명으로 "text-embedding-3-small"을 사용한다
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", disallowed_special=())

# CacaheBackedEmbedddings를 사용하여 임베딩 계산 결과를 캐시한다
# 이렇게 하면 임베딩을 여러번 계산할 필요 없이 한번 계산된 값을 재사용할 수 있다
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings, store, namespace=embeddings.model
)


########### Vector DB ##################

# langchain_community 모듈에서 FAISS 클래스를 가져옵니다
from langchain_community.vectorstores import FAISS

# 로컬에 저장할 FAISS 인덱스 폴더 이름을 저장한다
FAISS_DB_INDEX = "langchain_faiss"

# combined_documents 문서들과 cached_embeddings 임베딩을 사용하여
# FAISS 데이터 베이스를 생성합니다.
db = FAISS.from_documents(combined_documents, cached_embeddings)
db.save_local(folder_path=FAISS_DB_INDEX)



######## Vector DB 에서 꺼내는 코드

# langchain_community 모듈에서 FAISS 클래스를 가져온다
from langchain_community.vectorstores import FAISS

# FAISS 클래스의 load local 메서드를 사용하여 저장된 벡터 인덱스를 로드한다
db = FAISS.load_local(
    FAISS_DB_INDEX, # 로드할 FAISS 인덱스의 디렉토리 이름
    cached_embeddings, # 임베딩 정보를 제공
    allow_dangerous_deserialization=True, # 역질렬화를 허용하는 옵션
)




######## Retriever

# MMR을 사용하여 검색을 수행하는 retriever를 생성합니다.
faiss_retriever = db.as_retriever(serach_type="mmr", search_kwargs={"k":3})

# langchain.retrievers 모듈에서 BM25Retriever 클래스를 가져온다
from langchain_community.retrievers import BM25Retriever

# 문서 컬렉션을 사용하여 BM25 검색 모델 인스턴스를 생성한다
bm25_retriever = BM25Retriever.from_documents(
    combined_documents # 초기화에 사용할 문서 컬렉션
)

# BM25Retriever 인스턴스의 k 속성을 10으로 설정하여,
# 검색시 최대 10개의 결과를 반환하도록 한다
bm25_retriever.k = 3


# langchain.retievers 모듈에서 EnsembleRetriever 클래스를 가져온다
from langchain.retrievers import EnsembleRetriever

# EnsembleRetriever 인스턴스를 생성
# 이때, BM25 검색 모델과 FAISS 검색 모델을 결합하여 사용한다
ensemble_retriever = EnsembleRetriever(
    retrievers = [bm25_retriever, faiss_retriever], # 사용할 검색 모델의 리스트
    weights = [0.6, 0.4],
    search_type="mmr",
)


## Pipline

## Prompt
from langchain_core.prompts import PromptTemplate

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


rag_chain = (
    {"context": ensemble_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.with_config(configuarble={"llm":"gpt4"}).invoke(
    " RecursiveCharacterTextSplitter에 대해서 설명해줘"
)

