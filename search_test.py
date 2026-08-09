"""只测检索层：打印召回的段落，人工判断对不对。检索对了，生成才有意义。"""
import sys
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

vs = FAISS.load_local("faiss_index", OllamaEmbeddings(model="bge-m3"),
                      allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_kwargs={"k": 3})

q = sys.argv[1] if len(sys.argv) > 1 else "时区引擎是怎么实现的？"
for i, doc in enumerate(retriever.invoke(q)):
    print(f"--- hit {i+1} (from {doc.metadata.get('source')}) ---")
    print(doc.page_content[:200], "\n")