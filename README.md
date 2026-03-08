# TSG RAG SOP - 售服排查助手

这是一个基于 **RAG (Retrieval-Augmented Generation)** 技术的 AOI 机台售服排查工具。旨在通过语义检索过往的维修日誌，自动为工程师生成结构化的排查 SOP 流程。

## 🚀 核心功能

- **语义檢索 (Vector Search)**: 采用 `multilingual-MiniLM` 模型，支持中文/英文语义搜索，即使关键字不完全匹配（如「取像」与「影像通道」）也能找到关联案例。
- **模块化 SOP 生成**: 根据检索结果自动识别问题类别（如 `TRIGGER`, `CAPTURE`, `POWER_RISK`），并动态组立「通用流程 + 专属模块」的排查指南。
- **数据統計**: 实时汇总相似案例中的常见元件、问题分类与客户分佈，辅助根因判断。
- **本地化脫敏 (Data Anonymization)**: 内建脱敏工具，自动将客户名称与机台编号标签化，确保敏感数据不会上传至公有云或 GitHub。
- **Tkinter 桌面介面**: 直观的对话式 UI，方便现场工程师快速操作。

## 🛠️ 技术栈

- **Language**: Python 3.9+
- **Vector DB**: ChromaDB (Persistent Local Storage)
- **Embedding**: SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)
- **UI Framework**: Tkinter
- **Data Analysis**: Pandas

## 📁 专案结构

- `rag_ui_tk.py`: 主程式入口，提供 GUI 对话界面。
- `rag_engine.py`: RAG 核心逻辑，负责向量检索与 SOP 逻辑组合。
- `make_rag_docs.py`: 原始日誌预处理与**資料脫敏**脚本当。
- `build_chroma_db_local.py`: 构建/更新本地 Chroma 向量库。
- `anonymization_mapping.json`: (本地文件) 脱敏对照表，用于还原真实资讯。

## ⚙️ 快速开始

### 1. 安装依赖
建议使用虚拟环境：
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install pandas chromadb sentence-transformers
```

### 2. 資料預處理與脫敏
将您的原始日誌放在 `Combined_Maintenance_Log_Cleaned.csv`，然后运行：
```bash
python make_rag_docs.py
```
这会产生脱敏后的 `rag_docs.csv` 并生成本地对照表。

### 3. 构建向量资料库
```bash
python build_chroma_db_local.py
```

### 4. 启动排查助手
```bash
python rag_ui_tk.py
```

## ⚠️ 安全性说明
本专案已配置 `.gitignore` 以排除 `.xlsx`, `.csv` 与 `anonymization_mapping.json`。上傳至 GitHub 时，仅会保留脱敏后的 `rag_docs.csv` 与程式逻辑，确保商业资讯安全。
