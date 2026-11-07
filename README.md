# TSG RAG SOP - 售服排查助手

這是一個基於 **RAG (Retrieval-Augmented Generation)** 技術的 AOI 機台售服排查工具。旨在透過語義檢索過往的維修日誌，自動為工程師生成結構化的排查 SOP 流程。

## 🚀 核心功能

- **語義檢索 (Vector Search)**: 採用 `multilingual-MiniLM` 模型，支援中文/英文語義搜索，即使關鍵字不完全匹配（如「取像」與「影像通道」）也能找到關聯案例。
- **模組化 SOP 生成**: 根據檢索結果自動識別問題類別（如 `TRIGGER`, `CAPTURE`, `POWER_RISK`），並動態組建「通用流程 + 專屬模組」的排查指南。
- **數據統計**: 即時彙總相似案例中的常見元件、問題分類與客戶分佈，輔助根因判斷。
- **本地化脫敏 (Data Anonymization)**: 將客戶名稱與機台編號代號化，並以正則規則移除自由欄位中的 Email 與電話；輸出前會再稽核一次，有漏網即中止。
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
- `anonymization_mapping.json`: (本地檔案，不進版控) 脫敏對照表，另含 `extra_terms` 字詞黑名單。
- `rag_docs.sample.csv`: 結構示範用的**虛構**資料，內容與任何真實案例無關。

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
這會產生脫敏後的 `rag_docs.csv` 並生成本地對照表。該檔案不進版控；若只是想看資料長相或跑通流程，可直接使用 `rag_docs.sample.csv`。

### 3. 構建向量資料庫
```bash
python build_chroma_db_local.py
```

### 4. 啟動排查助手
```bash
python rag_ui_tk.py
```

## ⚠️ 安全性說明

**本倉庫不包含任何真實維修日誌。** 版控中只有程式邏輯與一份虛構的 `rag_docs.sample.csv`。

`.gitignore` 排除 `.xlsx`、`.csv`（`rag_docs.sample.csv` 除外）與 `anonymization_mapping.json`。

脫敏分兩層，缺一不可：

1. **欄位代號化**：`Client` → `Client_A`、`Machine_ID` → `M_xxxxx`（MD5 前 5 碼），對照表只存在本機。
2. **自由欄位遮蔽**：工程師常把客戶名、聯絡人與電話直接打在問題描述裡，這些值從未進過對照表，第 1 層抓不到。因此另以正則移除 Email／電話，並用 `extra_terms` 字詞黑名單處理公司名與人名。

`make_rag_docs.py` 在寫檔前會對輸出全文稽核一次，偵測到未遮蔽的聯絡資訊或黑名單字詞就中止，避免漏網資料被寫進可公開的檔案。
