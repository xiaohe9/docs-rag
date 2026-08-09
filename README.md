# 本地RAG问答系统

私有化部署的文档问答系统，基于 Ollama + Qwen3 + bge-m3 + FAISS。

## 技术栈
- **Embedding**: bge-m3（本地Ollama部署，中文向量化）
- **生成模型**: qwen3:4b（本地大模型，数据不出厂）
- **向量库**: FAISS（本地索引，零运维）
- **框架**: LangChain LCEL + FastAPI

## 运行
```bash
ollama pull bge-m3
ollama pull qwen3:4b
python ingest.py      # 建向量库
python rag.py "问题"   # 命令行问答
uvicorn main:app      # API服务
