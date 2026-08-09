from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

vs = FAISS.load_local("faiss_index", OllamaEmbeddings(model="bge-m3"),
                      allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 2})

# num_ctx=8192：Ollama默认2048，3个chunk+prompt容易爆
llm = ChatOllama(model="qwen3:4b", temperature=0.1, num_ctx=4096)

prompt = ChatPromptTemplate.from_template(
    "你是陈晓河项目文档的问答助手。仅基于以下资料回答问题，"
    "资料里没有就明确说「文档中没有提到」，不要编造。\n\n"
    "【资料】\n{context}\n\n【问题】{question}\n\n【回答】"
)

def fmt(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | fmt, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

def ask(q: str):
    answer = chain.invoke(q)
    sources = [d.metadata.get("source") for d in retriever.invoke(q)]
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "三层防幻觉是哪三层？"
    result = ask(q)
    print(f"【问题】{q}\n")
    print(f"【回答】{result['answer']}\n")
    print(f"【来源】{', '.join(set(result['sources']))}")