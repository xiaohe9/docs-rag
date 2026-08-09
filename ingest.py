"""把知识库文档切分、向量化，存进本地 FAISS 索引（分批版，防内存爆）"""
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 加载文档
docs = []
for path in glob.glob("knowledge/*.md"):
    docs.extend(TextLoader(path, encoding="utf-8").load())
print(f"loaded {len(docs)} docs")

# 2. 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"{len(docs)} docs -> {len(chunks)} chunks")

# 3. 分批向量化（每批20个，防Ollama崩溃）
emb = OllamaEmbeddings(model="bge-m3")
batch_size = 20

print(f"开始分批处理，每批{batch_size}个chunks...")
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    print(f"  处理第 {i+1}-{min(i+batch_size, len(chunks))} / {len(chunks)}...")
    
    if i == 0:
        vs = FAISS.from_documents(batch, emb)
    else:
        vs.add_documents(batch)

vs.save_local("faiss_index")
print("index saved to faiss_index/")