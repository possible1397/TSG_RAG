# query_rag_sop.py
# ============================================================
# 售服現場排查 SOP（Checklist 型 + RAG 輔助）
# - 固定 Checklist SOP（不被 log 敘事污染）
# - RAG 用來找相似案例 + TopK 統計（元件/分類/客戶）
# - 支援 where filter（若 metadata 乾淨）+ fallback 二階段聚焦（不怕 metadata 很亂）
# ============================================================

import os
import re
import sys
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ----------------------------
# 基本設定
# ----------------------------
CHROMA_DIR = "chroma_db_local"
COLLECTION_NAME = "maintenance_logs"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K_SHOW = 6          # 最後顯示幾筆案例
CANDIDATE_N = 30        # 先抓多少候選再做二階段篩選
MIN_GOOD_MATCH = 4      # 二階段後至少要留幾筆，少於就回退不篩選版本

# ----------------------------
# SOP 類型判斷
# ----------------------------
SOP_COUNT0 = "COUNT0"
SOP_CAPTURE_FAIL = "CAPTURE_FAIL"      # Count 正常但取像失敗
SOP_POWER_TRIP_BURN = "POWER_TRIP_BURN"
SOP_GENERAL = "GENERAL"

def detect_sop_type(query: str) -> str:
    q = (query or "").strip().lower()

    # 跳電/燒卡優先（高風險）
    if any(k in q for k in ["跳電", "燒", "燒毀", "焦味", "冒煙", "短路", "dc24", "24v"]):
        return SOP_POWER_TRIP_BURN

    # Count0
    if "count 0" in q or "count0" in q or "計數為0" in q or "計數=0" in q or "count=0" in q:
        return SOP_COUNT0

    # 取像失敗類
    if any(k in q for k in ["取像失敗", "count正常", "count 有值", "有count", "有值但取像失敗", "有值但無圖", "有值但影像"]):
        return SOP_CAPTURE_FAIL

    # Trigger/訊號不穩，也多半走 Count0/訊號類，但保守先 GENERAL，避免誤導
    if any(k in q for k in ["trigger", "觸發", "訊號不穩", "雜訊", "干擾"]):
        return SOP_GENERAL

    return SOP_GENERAL


# ----------------------------
# 固定 Checklist SOP 模板
# ----------------------------
def get_checklist_text(sop_type: str) -> str:
    if sop_type == SOP_COUNT0:
        return """------------------------------------------------------------------------
【Count 0 / Trigger 無觸發】排查 Checklist
------------------------------------------------------------------------
目標：先確認『Trigger 訊號有沒有到取像端』，再定位是接線/訊號/設定/卡件問題。

A) 10 分鐘內先做（快速定位）
  ☐ 1. 看 Log / Counter：確認是 Count=0（還是 Count 有值但取像失敗）。
  ☐ 2. 用 DAQ/示波器量 Trigger 訊號（靠近取像卡/Trigger 卡輸入端）：確認有無脈波、幅度/寬度/抖動是否正常。
  ☐ 3. 核對 I/O 表與接線：Trigger 線是否接到正確腳位/Channel（避免接到虛擬軸或錯誤 IO）。
  ☐ 4. 檢查 LineScan A/B port 是否接反（常見）：背面 OK / 正面 NG 或換線後突然 NG。
  ☐ 5. 確認相機 TriggerMode：是否為外部觸發 (External Trigger)，是否被改回 FreeRun。

B) 30–60 分鐘（深入查修）
  ☐ 1. 確認觸發極性/模式：High/Low Active 是否與實際訊號一致（示波器確認有效脈波方向）。
  ☐ 2. 檢查 OVP 卡是否造成 Trigger 訊號被濾掉/減少：必要時依案例改用對應訊號腳（例如 CMP-/GND）並同步調整 Active polarity。
  ☐ 3. 檢查訊號品質與接地：屏蔽線/隔離網/接地線是否確實，避免地迴路與電源雜訊耦合。
  ☐ 4. 檢查取像卡設定檔（mcf）：Active polarity、threshold、filter/debounce 是否需因應訊號路徑調整。

C) 必要時（交叉驗證 / 收斂到硬體或外部因素）
  ☐ 1. 互換法：Trigger 卡/OVP 卡/取像卡換到 OK 機台（或反向互換）→ 確認是否卡件故障。
  ☐ 2. 隔離法定位：拔除可疑 sensor/外部 I/O 支路逐一測試，看 Count 0 是否消失（定位短路/破皮/干擾源）。
  ☐ 3. 修正後回歸測試：連續跑一定片數/一定時間，確認 Count/取像穩定並記錄變更（接線/設定/更換料號）。

✅ 何時升級工程師（符合任一條就可以打電話）
  - 1) DAQ/示波器完全量不到 Trigger 脈波，但接線已確認正確。
  - 2) 已確認外部觸發 + Active polarity 正確，仍 Count 0。
  - 3) 發現跳電紀錄/燒卡痕跡/焦味（避免擴大損壞）。
  - 4) 互換法確認卡件異常但現場無備品。
"""
    if sop_type == SOP_CAPTURE_FAIL:
        return """------------------------------------------------------------------------
【取像失敗（Count 正常）】排查 Checklist
------------------------------------------------------------------------
目標：Count 有值代表 Trigger 可能正常，改查取像流程/設定/資料流/線材/相機輸出。

A) 10 分鐘內先做（快速定位）
  ☐ 1. 確認 Count 是否足夠：不是 0，且穩定累加。
  ☐ 2. 檢查相機輸出：曝光/增益是否異常；是否全黑/過曝。
  ☐ 3. 檢查線材接觸不良：CameraLink/網路線重新插拔；可先換線快速驗證。
  ☐ 4. 重啟取像流程：重啟軟體/FrameGrabber；確認 Driver 初始化成功。

B) 30–60 分鐘（深入查修）
  ☐ 1. 檢查取像卡設定檔（mcf）/Buffer：是否套錯檔、參數與現場不符、buffer overflow。
  ☐ 2. 檢查同步/時序：觸發脈波寬度是否符合相機需求；曝光與觸發間隔是否過短。
  ☐ 3. 檢查軟體版本/權限/防毒：更新後是否導致取像被阻擋或權限不足。

C) 必要時（交叉驗證）
  ☐ 1. 互換線材/相機：確認是否為相機或線材故障。
  ☐ 2. 互換取像卡/電腦：確認是否為卡件或主機問題。

✅ 何時升級工程師
  - 1) Count 正常但一直取像失敗，重啟/換線無效
  - 2) 需要確認程式/Driver/版本差異、或要看更深層 log
"""
    if sop_type == SOP_POWER_TRIP_BURN:
        return """------------------------------------------------------------------------
【跳電 / 燒卡 / 焦味】高風險排查 Checklist
------------------------------------------------------------------------
目標：先保護設備，定位跳電支路/短路點，避免反覆上電造成二次損壞。

A) 立即措施（1–3 分鐘）
  ☐ 1. 停止操作、避免反覆上電測試。
  ☐ 2. 記錄現象：何時跳電、哪個動作觸發、是否伴隨焦味/冒煙/燒痕。

B) 10–30 分鐘（定位短路/支路）
  ☐ 1. 查 DC24V / AC 供電與跳電紀錄：保險絲/電源供應器/端子台是否發熱或燒痕。
  ☐ 2. 沿 DC24V 支路查破皮/短路：sensor 線、活動機構附近線材、清潔站離子槍回路等高風險區。
  ☐ 3. 隔離法：拔除可疑支路（sensor/致動器/外部 I/O）逐一測試，看跳電是否消失。

C) 必要時（卡件處置/驗證）
  ☐ 1. 卡件目檢：Trigger/OVP/取像卡是否有燒痕、變色、焦味（必要時顯微鏡檢焊點/元件）。
  ☐ 2. 若有燒痕：避免繼續通電，改以備品更換並回收送修分析。
  ☐ 3. 修正後回歸測試：長時間空跑確認無跳電，再恢復生產。

✅ 何時升級工程師
  - 1) 確認有燒痕/焦味/冒煙
  - 2) 無法定位短路支路
  - 3) 換卡後仍跳電（外部問題未解）
"""
    # GENERAL
    return """------------------------------------------------------------------------
【通用排查 Checklist】
------------------------------------------------------------------------
目標：先釐清是 Trigger/訊號、取像流程、還是跳電/硬體損壞，再依類型切換 SOP。

A) 先釐清（30 秒）
  ☐ 1. 問題是 Count 0？還是 Count 正常但取像失敗？
  ☐ 2. 一直發生還是偶發？發生在上電/掃像/切換模式/換料後？
  ☐ 3. 近期是否改線/換卡/換 sensor/更新軟體？

B) 快速動作（10 分鐘）
  ☐ 1. 看 Log/Counter + 量 Trigger（DAQ/示波器）。
  ☐ 2. 核對接線/I-O 表、A/B port、TriggerMode。
  ☐ 3. 如有跳電/燒痕：改走「跳電/燒卡」SOP。
"""


# ----------------------------
# metadata 探測（不確定前人寫法時用）
# ----------------------------
def try_probe_metadata_values(collection, field: str, sample_n: int = 50):
    """
    嘗試抓 metadata 的 field 值分布（若取不到/沒有欄位就回 None）。
    Chroma 版本不同，get() 參數可能不同，因此做容錯。
    """
    try:
        got = collection.get(limit=sample_n, include=["metadatas"])
        metas = got.get("metadatas", []) or []
        vals = []
        for md in metas:
            if isinstance(md, dict) and field in md:
                v = md.get(field, "")
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    vals.append(s)
        if not vals:
            return None
        s = pd.Series(vals)
        return s.value_counts().head(10).to_dict()
    except Exception:
        return None


# ----------------------------
# 二階段聚焦：用 SOP 類型做內容關鍵字篩選 / re-rank
# ----------------------------
def sop_relevant_score(sop_type: str, text: str) -> int:
    t = (text or "").lower()

    if sop_type == SOP_COUNT0:
        # Count0/trigger 相關詞
        keys = [
            "count 0", "count=0", "count0", "計數為 0", "計數為0",
            "trigger", "觸發", "取像失敗", "daq", "mcf", "ovp", "cmp", "a/b", "port"
        ]
    elif sop_type == SOP_CAPTURE_FAIL:
        keys = ["取像失敗", "count", "曝光", "全黑", "過曝", "buffer", "driver", "mcf", "fram", "frame"]
    elif sop_type == SOP_POWER_TRIP_BURN:
        keys = ["跳電", "24v", "dc24", "短路", "破皮", "燒", "燒毀", "焦味", "保險絲", "sensor", "離子槍"]
    else:
        keys = ["取像", "count", "trigger", "跳電", "ovp", "mcf", "driver", "daq"]

    score = 0
    for k in keys:
        if k in t:
            score += 1
    return score


def filter_and_rerank_by_sop(sop_type: str, candidates: list):
    """
    candidates: list of dict {id, doc, md, dist}
    依 SOP 類型打分並篩掉明顯不相關者，再依 (score desc, dist asc) 排序。
    """
    scored = []
    for c in candidates:
        doc = c.get("doc", "")
        md = c.get("md", {}) or {}
        # content + 一點 metadata 一起評分更穩
        blob = doc + "\n" + " ".join([f"{k}:{v}" for k, v in md.items() if v is not None])
        s = sop_relevant_score(sop_type, blob)
        c["sop_score"] = s
        scored.append(c)

    # 篩掉 score=0 的（太離題）
    filtered = [c for c in scored if c["sop_score"] > 0]

    # 如果篩完太少，就退回不篩選（避免沒結果）
    if len(filtered) < MIN_GOOD_MATCH:
        filtered = scored

    filtered.sort(key=lambda x: (-x["sop_score"], x.get("dist", 9999)))
    return filtered


# ----------------------------
# 顯示用：抽元件/分類/客戶統計
# ----------------------------
def extract_component_list(md: dict) -> str:
    # 你 metadata 裡可能是 key_component，也可能叫 Key_Component
    for k in ["key_component", "Key_Component"]:
        if k in md and md[k]:
            return str(md[k])
    return ""

def extract_category(md: dict) -> str:
    for k in ["problem_category", "Problem_Category"]:
        if k in md and md[k]:
            return str(md[k])
    return ""

def extract_client(md: dict) -> str:
    for k in ["client", "Client"]:
        if k in md and md[k]:
            return str(md[k])
    return ""

def extract_date(md: dict) -> str:
    for k in ["date", "Date"]:
        if k in md and md[k]:
            return str(md[k])
    return ""

def extract_machine(md: dict) -> str:
    for k in ["machine_id", "Machine_ID"]:
        if k in md and md[k]:
            return str(md[k])
    return ""

def short_summary_from_doc(doc: str) -> str:
    """
    只抽對售服有用的幾行：Symptom/原因/跳電原因/更換硬體/維修處理
    你的 content 以 [Full_Content] 開始，因此直接用 regex 抓關鍵句。
    """
    s = doc or ""
    # 優先找 Full_Content 區塊
    m = re.search(r"\[Full_Content\]\s*(.+)", s, flags=re.S)
    body = m.group(1) if m else s

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    keep = []

    # 抓含關鍵字的行（保守）
    kw = ["問題現象", "跳電原因", "原因", "查修", "維修處理", "更換硬體", "接線", "設定", "Count", "count", "Trigger", "OVP", "CMP"]
    for ln in lines:
        if any(k in ln for k in kw):
            keep.append(ln)
        if len(keep) >= 6:
            break

    if not keep:
        # fallback: 前 2 行
        keep = lines[:2]

    return " / ".join(keep)[:220]  # 截斷避免太長


# ----------------------------
# 查詢 + 輸出
# ----------------------------
def query_chroma(collection, model, question: str, where: dict | None, n: int):
    q_emb = model.encode([question], normalize_embeddings=True).tolist()[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances", "ids"],
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for i in range(len(ids)):
        out.append({
            "id": ids[i],
            "doc": docs[i] if i < len(docs) else "",
            "md": metas[i] if i < len(metas) else {},
            "dist": float(dists[i]) if i < len(dists) else 9999.0,
        })
    return out


def print_stats(cands: list):
    comp_counter = {}
    cat_counter = {}
    client_counter = {}

    for c in cands:
        md = c.get("md", {}) or {}
        comps = extract_component_list(md)
        cat = extract_category(md)
        cli = extract_client(md)

        # 元件可能是 "A, B, C" 字串
        if comps:
            for x in [p.strip() for p in comps.split(",") if p.strip()]:
                comp_counter[x] = comp_counter.get(x, 0) + 1
        if cat:
            cat_counter[cat] = cat_counter.get(cat, 0) + 1
        if cli:
            client_counter[cli] = client_counter.get(cli, 0) + 1

    def topk(d, k=5):
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
        return ", ".join([f"{a}({b})" for a, b in items]) if items else "N/A"

    print("常見關聯元件：", topk(comp_counter, 8))
    print("常見問題分類：", topk(cat_counter, 6))
    print("客戶分布（參考）：", topk(client_counter, 8))


def main():
    print("載入 embedding 模型：", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    # 探測 metadata（可選：讓你知道前人到底寫什麼）
    # 不影響功能，只是讓你能印出來看
    probe = try_probe_metadata_values(col, "problem_category", sample_n=50)
    if probe:
        print("（探測）problem_category 可能值 Top10：", probe)

    while True:
        q = input("\n請輸入你的問題（輸入 q 離開）：").strip()
        if q.lower() in ["q", "quit", "exit"]:
            break

        sop_type = detect_sop_type(q)

        print("\n" + "=" * 72)
        print("🧰 售服現場排查 SOP（Checklist 型 + RAG 輔助）")
        print("=" * 72)
        print(f"問題：{q}")
        print(f"SOP 類型判斷：{sop_type}\n")

        # 1) 固定 checklist
        print(get_checklist_text(sop_type))

        # 2) RAG 相似案例（先嘗試 where filter，失敗就 where=None）
        #   - 你不確定 metadata 值，所以 where 先保守
        where = None

        # 如果你未來確認 problem_category 很乾淨，可以解鎖下面這段：
        # if sop_type in [SOP_COUNT0, SOP_POWER_TRIP_BURN]:
        #     where = {"problem_category": "訊號與干擾"}

        # 先抓候選
        try:
            candidates = query_chroma(col, model, q, where=where, n=CANDIDATE_N)
        except Exception:
            candidates = query_chroma(col, model, q, where=None, n=CANDIDATE_N)

        # 二階段：依 SOP 類型做聚焦（不靠 metadata 也能收斂）
        focused = filter_and_rerank_by_sop(sop_type, candidates)
        show = focused[:TOP_K_SHOW]

        print("------------------------------------------------------------------------")
        print("📌 RAG 相似案例彙整（用來增加可信度 / 指出常見方向）")
        print("------------------------------------------------------------------------")
        print_stats(show)
        print("\n5) 參考案例（Top Matches）\n")

        for i, c in enumerate(show, 1):
            md = c.get("md", {}) or {}
            doc_id = c.get("id", "")
            dist = c.get("dist", 9999.0)

            date = extract_date(md)
            cli = extract_client(md)
            mach = extract_machine(md)
            cat = extract_category(md)
            comp = extract_component_list(md)

            summary = short_summary_from_doc(c.get("doc", ""))

            print(f"  [{i}] {date} | {cli} | {mach} | {cat} | {comp}")
            print(f"      doc_id={doc_id}")
            print(f"      distance={dist:.4f}  sop_score={c.get('sop_score', 0)}")
            print(f"      摘要： {summary}\n")

        print("✅ SOP 輸出完成（Checklist 型）")
        print("   - SOP 是固定流程，避免 log 原始敘事導致『原因/對策混寫』")
        print("   - RAG 只用來提供相似案例與常見元件方向，售服照 checklist 做即可")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)
