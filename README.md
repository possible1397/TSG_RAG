# TSG RAG SOP - 售服排查助手

這是一個基於 **RAG (Retrieval-Augmented Generation)** 技術的 AOI 機台售服排查工具。旨在透過語義檢索過往的維修日誌，自動為工程師生成結構化的排查 SOP 流程。

## 🚀 核心功能

- **語義檢索 (Vector Search)**: 採用 `multilingual-MiniLM` 模型，支援中文/英文語義搜索，即使關鍵字不完全匹配（如「取像」與「影像通道」）也能找到關聯案例。
- **模組化 SOP 生成**: 根據檢索結果自動識別問題類別（如 `TRIGGER`, `CAPTURE`, `POWER_RISK`），並動態組建「通用流程 + 專屬模組」的排查指南。
- **數據統計**: 即時彙總相似案例中的常見元件、問題分類與客戶分佈，輔助根因判斷。
- **本地化脫敏 (Data Anonymization)**: 內建脫敏工具，自動將客戶名稱與機台編號標籤化，確保敏感數據不會上傳至公有雲或 GitHub。
- **Tkinter 桌面介面**: 直觀的對話式 UI，方便現場工程師快速操作。

## 🛠️ 技術棧

- **Language**: Python 3.9+
- **Vector DB**: ChromaDB (Persistent Local Storage)
- **Embedding**: SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)
- **UI Framework**: Tkinter
- **Data Analysis**: Pandas

## 📁 專案結構

- `rag_ui_tk.py`: 主程式入口，提供 GUI 對話介面。
- `rag_engine.py`: RAG 核心邏輯，負責向量檢索與 SOP 邏輯組合。
- `make_rag_docs.py`: 原始日誌預處理與**資料脫敏**腳本。
- `build_chroma_db_local.py`: 構建/更新本地 Chroma 向量庫。
- `anonymization_mapping.json`: (本地檔案) 脫敏對照表，用於還原真實資訊。

## ⚙️ 快速開始

### 1. 安裝依賴
建議使用虛擬環境：
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install pandas chromadb sentence-transformers
```

### 2. 資料預處理與脫敏
將您的原始日誌放在 `Combined_Maintenance_Log_Cleaned.csv`，然後執行：
```bash
python make_rag_docs.py
```
這會產生脫敏後的 `rag_docs.csv` 並生成本地對照表。

### 3. 構建向量資料庫
```bash
python build_chroma_db_local.py
```

### 4. 啟動排查助手
```bash
python rag_ui_tk.py
```

## ⚠️ 安全性說明
本專案已配置 `.gitignore` 以排除 `.xlsx`, `.csv` 與 `anonymization_mapping.json`。上傳至 GitHub 時，僅會保留脫敏後的 `rag_docs.csv` 與程式邏輯，確保商業資訊安全。
