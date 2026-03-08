import pandas as pd
import re
import os
import json
import hashlib

INPUT_CSV = "Combined_Maintenance_Log_Cleaned.csv"
OUTPUT_CSV = "rag_docs.csv"
MAPPING_FILE = "anonymization_mapping.json"

# 針對 Description 的垃圾列判斷
_BAD_DESC_PATTERNS = [
    r"^[:：]+$",
    r"^[-–—]+$",
    r"^\.+$",
    r"^,+$",
    r"^;$",
]


class Anonymizer:
    def __init__(self, mapping_file=MAPPING_FILE):
        self.mapping_file = mapping_file
        self.client_map = {}
        self.machine_map = {}
        self._load_mapping()

    def _load_mapping(self):
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.client_map = data.get("clients", {})
                    self.machine_map = data.get("machines", {})
            except Exception as e:
                print(f"警告：讀取對照表失敗 ({e})，將啟動全新對照。")

    def _save_mapping(self):
        with open(self.mapping_file, "w", encoding="utf-8") as f:
            json.dump({
                "clients": self.client_map,
                "machines": self.machine_map
            }, f, indent=4, ensure_ascii=False)

    def get_client_alias(self, name: str) -> str:
        if not name or name.lower() == "nan": return "Unknown_Client"
        name = str(name).strip()
        if name not in self.client_map:
            alias = f"Client_{chr(65 + len(self.client_map))}" if len(self.client_map) < 26 else f"Client_{len(self.client_map)}"
            self.client_map[name] = alias
        return self.client_map[name]

    def get_machine_alias(self, machine_id: str) -> str:
        if not machine_id or machine_id.lower() == "nan": return "M_Unknown"
        machine_id = str(machine_id).strip()
        if machine_id not in self.machine_map:
            # 使用 Hash 產生簡短代碼
            h = hashlib.md5(machine_id.encode()).hexdigest()[:5].upper()
            alias = f"M_{h}"
            self.machine_map[machine_id] = alias
        return self.machine_map[machine_id]


def is_trivial_text(s: str) -> bool:
    if s is None:
        return True
    t = str(s).strip()
    if t == "" or t.lower() == "nan":
        return True
    if len(t) < 5:
        return True
    for p in _BAD_DESC_PATTERNS:
        if re.match(p, t):
            return True
    return False


def safe_str(v) -> str:
    if v is None:
        return ""
    s = str(v)
    if s.lower() == "nan":
        return ""
    return s.strip()


def build_content(row: pd.Series, anonymizer: Anonymizer) -> str:
    # 內容（embedding 用）：把分類、元件等「寫進內容」提升本地 embedding 準確度
    date = safe_str(row.get("Date", ""))
    
    # 脫敏處理
    client = anonymizer.get_client_alias(row.get("Client", ""))
    machine_id = anonymizer.get_machine_alias(row.get("Machine_ID", ""))

    cat = safe_str(row.get("Problem_Category", ""))
    comp = safe_str(row.get("Key_Component", ""))

    symptom = safe_str(row.get("Symptom", ""))
    desc = safe_str(row.get("Description", ""))
    result = safe_str(row.get("Result", ""))
    hardware = safe_str(row.get("Hardware", ""))
    root = safe_str(row.get("Root_Cause", ""))

    source = safe_str(row.get("Source_File", ""))
    full_content = safe_str(row.get("Full_Content", ""))

    # 對 Full_Content 進行簡單的關鍵字替換（若內容中包含原始 Client/Machine_ID）
    for raw_name, alias in anonymizer.client_map.items():
        if len(raw_name) > 2: # 避免誤殺太短的字
            full_content = full_content.replace(raw_name, alias)
    for raw_id, alias in anonymizer.machine_map.items():
        full_content = full_content.replace(raw_id, alias)

    parts = [
        "[Type] MaintenanceLog",
        f"[Date] {date}",
        f"[Client] {client}",
        f"[Machine_ID] {machine_id}",
        "",
        f"[Problem_Category] {cat}",
        f"[Key_Component] {comp}",
        "",
        f"[Symptom] {symptom}",
    ]

    if full_content:
        parts += ["", "[Full_Content]", full_content]
    else:
        parts += [
            f"[Description] {desc}",
            f"[Result/Solution] {result}",
            f"[Hardware_Change] {hardware}",
            f"[Root_Cause] {root}",
        ]

    if source:
        parts += ["", f"[Source] {source}"]

    content = "\n".join([p for p in parts if p is not None])
    return content.strip()


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"找不到 {INPUT_CSV}，請確認檔案在同一資料夾。")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    anonymizer = Anonymizer()

    # 缺欄補空
    for col in [
        "Date", "Client", "Machine_ID", "Problem_Category", "Key_Component",
        "Symptom", "Root_Cause", "Full_Content", "Source_File",
        "Description", "Result", "Hardware"
    ]:
        if col not in df.columns:
            df[col] = ""

    # 產生 row_index
    df = df.reset_index(drop=True)
    df["row_index"] = df.index.astype(int)

    # 建 content (包含脫敏)
    df["content"] = df.apply(lambda r: build_content(r, anonymizer), axis=1)

    # 對 metadata 欄位脫敏
    df["date"] = df["Date"].astype(str)
    df["year"] = df["date"].str.extract(r"^(\d{4})", expand=False).fillna("")
    df["client"] = df["Client"].apply(anonymizer.get_client_alias)
    df["machine_id"] = df["Machine_ID"].apply(anonymizer.get_machine_alias)
    df["problem_category"] = df["Problem_Category"].astype(str).fillna("")
    df["key_component"] = df["Key_Component"].astype(str).fillna("")
    df["source_file"] = df["Source_File"].astype(str).fillna("")
    df["has_result"] = df["Full_Content"].astype(str).str.contains(r"維修處理|處理結果|Result", regex=True, na=False)
    df["has_hardware_change"] = df["Full_Content"].astype(str).str.contains(r"更換硬體|硬體|Hardware", regex=True, na=False)

    # 建 doc_id (使用脫敏後的 machine_id)
    def make_doc_id(r):
        sf = safe_str(r.get("Source_File", ""))
        mi = safe_str(r.get("machine_id", ""))
        d = safe_str(r.get("Date", ""))
        ri = int(r.get("row_index", -1))
        return f"{sf}::{ri}::{mi}::{d}"

    df["doc_id"] = df.apply(make_doc_id, axis=1)

    # 垃圾列過濾
    df["content_len"] = df["content"].astype(str).str.len()
    df = df[df["content_len"] >= 60].copy()

    out_cols = [
        "doc_id",
        "content",
        "date", "year", "client", "machine_id", "problem_category",
        "key_component", "source_file", "row_index",
        "has_result", "has_hardware_change",
    ]

    rag_df = df[out_cols].copy()
    rag_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    anonymizer._save_mapping()

    print("✅ 已產生 RAG 脫敏資料：", OUTPUT_CSV)
    print("✅ 已更新對照表：", MAPPING_FILE)
    print("總筆數：", len(rag_df))
    print(rag_df[["doc_id", "date", "client", "machine_id"]].head(5))


if __name__ == "__main__":
    main()
