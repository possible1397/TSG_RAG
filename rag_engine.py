import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import Counter

CHROMA_DIR = "chroma_db_local"
COLLECTION_NAME = "maintenance_logs"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =========================
# 1) 問句粗分類（保留：可當初始提示用）
#    注意：這不會限制你只能問三種，真正 SOP 會由 RAG matches 推 tags
# =========================
def classify_sop_type(question: str) -> str:
    q = (question or "").lower()
    if "count 0" in q or "count0" in q:
        return "COUNT0"
    if "跳電" in (question or "") or "燒" in (question or "") or "燒毀" in (question or ""):
        return "POWER_TRIP_BURN"
    if "取像失敗" in (question or ""):
        return "CAPTURE_FAIL"
    if "ovp" in q:
        return "OVP_RELATED"
    return "GENERAL"


# =========================
# 2) 工具：metadata 統計用
# =========================
def _parse_components(s: str) -> list[str]:
    if not s:
        return []
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p and p != "N/A"]


def _top_counter(items: list[str], topn=8) -> str:
    # 過濾掉 Unknown 或 N/A 的統計
    filtered = [it for it in items if it and it not in ["N/A", "Unknown_Client", "M_Unknown"]]
    c = Counter(filtered)
    return ", ".join([f"{k}({v})" for k, v in c.most_common(topn)]) if c else "N/A"


# =========================
# 3) 新增 #1：從 RAG matches 推論 tags（可持續擴充）
#    這裡是「可擴充點」：你以後遇到新問題類型，只要加規則即可
# =========================
def infer_tags_from_matches(matches: list[dict]) -> list[str]:
    """
    從 RAG 撈回來的 matches（content/metadata）推論問題標籤 tags。
    tags 不是 sop_type（三選一），而是可疊加、多種模組同時啟用。
    """
    text = " ".join([
        (m.get("content") or "") + " " +
        (m.get("problem_category") or "") + " " +
        (m.get("key_component") or "")
        for m in matches
    ]).lower()

    tags = set(["GENERAL"])

    # Trigger / Count0 類
    if "count 0" in text or "count0" in text or "trigger" in text or "觸發" in text:
        tags.update(["TRIGGER"])

    # Capture / 影像通道 類
    if "取像" in text or "cameralink" in text or "framegrabber" in text or "grab" in text or "image" in text:
        tags.update(["CAPTURE"])

    # Power Risk 類：跳電/短路/燒毀
    if "跳電" in text or "短路" in text or "燒" in text or "焦味" in text or "burn" in text:
        tags.update(["POWER_RISK"])

    # OVP 類
    if "ovp" in text:
        tags.update(["OVP"])

    # 光源類
    if "光源" in text or "照明" in text or "lighting" in text or "strobe" in text:
        tags.update(["LIGHTING"])

    # 機構/伺服/原點類
    if "馬達" in text or "servo" in text or "原點" in text or "home" in text or "限位" in text or "軸" in text:
        tags.update(["MOTION"])

    # 軟體/版本/driver 類
    if "driver" in text or "版本" in text or "更新" in text or "dll" in text or "exception" in text or "error" in text:
        tags.update(["SOFTWARE"])

    # 網路/通訊 類
    if "timeout" in text or "socket" in text or "網路" in text or "ping" in text or "連線" in text:
        tags.update(["NETWORK"])

    return sorted(tags)


# =========================
# 4) 新增 #2：依 tags 組合 SOP（通用 + 模組化）
#    最後一步固定：✅ 回報工程師
# =========================
def build_modular_sop(tags: list[str]) -> str:
    steps: list[str] = []

    # ---- 通用流程（永遠有） ----
    steps += [
        "確認現象與復現條件（固定/間歇、哪一站/哪一顆相機/哪一支路、是否近期換線/改參數/更新版本）",
        "保存現場資訊（Log/截圖/時間點/機台編號/站別/相機通道/工單資訊）",
        "確認是否有高風險狀態（跳電/焦味/燒痕/異常發熱）→ 有則先停機止損並避免反覆上電",
        "核對接線與設定是否正確（I/O 表、接線腳位、port、設定檔是否套錯或被更動）",
        "互換/隔離法定位問題源（線材/卡件/外部支路逐一排除，找出『一換就好』或『一接回就壞』的關鍵）",
        "修正後回歸測試並記錄變更內容（做了什麼、換了什麼、參數改哪裡、測了多久/幾片）",
    ]

    # ---- 依 tags 插入模組 ----
    if "TRIGGER" in tags:
        steps += [
            "量測 Trigger 訊號（取像端輸入）確認脈波是否存在、幅度/寬度/頻率/抖動是否正常",
            "確認 TriggerMode 與 polarity（External Trigger / Active High-Low）是否與實測波形一致",
            "檢查取像卡/設定檔（threshold / debounce / filter）是否把脈波濾掉或門檻過高",
            "檢查線材與 Port（LineScan A/B 是否接反、接頭鬆脫、針腳歪斜、線材內斷）",
        ]

    if "CAPTURE" in tags:
        steps += [
            "檢查影像狀態（全黑/過曝/破圖/延遲/掉幀），先排除曝光/增益/光源異常",
            "重新插拔相機線材與接頭（CameraLink/網路線/供電），並重啟相機與取像程式",
            "核對取像卡設定檔（mcf）與影像通道（頻寬/丟包/PC 資源滿載）",
        ]

    if "OVP" in tags:
        steps += [
            "若訊號經過 OVP/保護板/中介板：確認接法與參考地正確，必要時調整腳位與 polarity 做交叉驗證",
        ]

    if "POWER_RISK" in tags:
        steps += [
            "以隔離法定位短路支路（拔除外部 I/O 支路，逐一接回找出『一接就跳』的支路）",
            "檢查線材破皮/夾傷/磨損（拖鏈/門鏈/活動機構/清潔站附近優先），以及端子台熔損",
            "確認接地/屏蔽避免地迴路與雜訊耦合，修復後再進行回歸測試",
        ]

    if "LIGHTING" in tags:
        steps += [
            "檢查光源供電、亮度設定、strobe 同步觸發與接線，確認光源是否間歇熄滅或不穩",
        ]

    if "MOTION" in tags:
        steps += [
            "檢查原點/限位/伺服告警與 I/O 狀態，確認機構是否卡滯、干涉、或回原點流程異常",
        ]

    if "SOFTWARE" in tags:
        steps += [
            "確認最近版本更新/驅動/防毒/權限是否影響，必要時比對 OK 機台版本或回退版本驗證",
        ]

    if "NETWORK" in tags:
        steps += [
            "檢查網路連線品質（ping/丟包/交換器/線材），確認 timeout 是否為通訊不穩或伺服端問題",
        ]

    # ---- 固定最後一步 ----
    steps.append("✅ 回報工程師（附上：現象描述、log/截圖、量測結果、已排除項目、互換/隔離結論）")

    # ---- 輸出格式 ----
    out = [
        "------------------------------------------------------------------------",
        "【售服問題排查流程 SOP（通用 + 模組化）】",
        "------------------------------------------------------------------------",
        f"啟用模組 tags: {', '.join(tags) if tags else 'GENERAL'}",
        ""
    ]
    for i, s in enumerate(steps, 1):
        out.append(f"{i}) {s}")
    return "\n".join(out).rstrip()


class RAGEngine:
    def __init__(self, top_k: int = 6):
        self.top_k = top_k
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def ask(self, question: str) -> dict:
        sop_type = classify_sop_type(question)

        # 防呆：DB 空
        if self.col.count() == 0:
            tags = ["GENERAL"]
            sop_text = build_modular_sop(tags)
            return {
                "question": question,
                "sop_type": sop_type,
                "tags": tags,
                "sop_text": sop_text,
                "stats": {"components": "N/A", "categories": "N/A", "clients": "N/A"},
                "matches": [],
                "error": "Chroma collection is empty. Please ingest documents first."
            }

        # 查詢 embedding
        q_emb = self.model.encode([question], normalize_embeddings=True).tolist()

        # ✅ include 不要放 "ids"（ids 會自動回傳）
        res = self.col.query(
            query_embeddings=q_emb,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        mds = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        matches = []
        for doc_id, content, md, dist in zip(ids, docs, mds, dists):
            md = md or {}
            matches.append({
                "doc_id": doc_id,
                "distance": float(dist) if dist is not None else None,
                "date": md.get("date", ""),
                "client": md.get("client", ""),
                "machine_id": md.get("machine_id", ""),
                "problem_category": md.get("problem_category", ""),
                "key_component": md.get("key_component", ""),
                "preview": (content or "").replace("\n", " ")[:220],
                "content": content or ""
            })

        # 統計
        all_components = []
        all_categories = []
        all_clients = []
        for m in matches:
            all_components += _parse_components(m.get("key_component", ""))
            if m.get("problem_category"):
                all_categories.append(m["problem_category"])
            if m.get("client"):
                all_clients.append(m["client"])

        stats = {
            "components": _top_counter(all_components, 8),
            "categories": _top_counter(all_categories, 6),
            "clients": _top_counter(all_clients, 6),
        }

        # ✅ 真正 SOP：由 RAG matches 推 tags，再組「通用 + 模組化」流程
        tags = infer_tags_from_matches(matches)
        sop_text = build_modular_sop(tags)

        return {
            "question": question,
            "sop_type": sop_type,   # 保留：當作初始粗分類
            "tags": tags,           # 新增：真正啟用的模組
            "sop_text": sop_text,
            "stats": stats,
            "matches": matches
        }
