import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

RAG_CSV = "rag_docs.csv"
CHROMA_DIR = "chroma_db_local"
COLLECTION_NAME = "maintenance_logs"

# 本地 embedding 模型（CPU 友好、中文可用）
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    if not os.path.exists(RAG_CSV):
        raise FileNotFoundError(f"找不到 {RAG_CSV}，請先跑 make_rag_docs.py")

    df = pd.read_csv(RAG_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=["doc_id", "content"]).copy()

    # 初始化 embedding 模型
    print("載入 embedding 模型：", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 初始化 Chroma（本地持久化）
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # 若已存在同名 collection，直接取用（避免每次重建）
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    # 準備資料
    ids = df["doc_id"].astype(str).tolist()
    documents = df["content"].astype(str).tolist()

    # metadata（Chroma 需要 dict）
    meta_cols = [
        "date", "year", "client", "machine_id", "problem_category",
        "key_component", "source_file", "row_index",
        "has_result", "has_hardware_change",
    ]

    metadatas = []
    for _, r in df.iterrows():
        md = {}
        for c in meta_cols:
            v = r.get(c, "")
            # pandas bool/numpy 型別轉 python bool
            if isinstance(v, (bool,)):
                md[c] = bool(v)
            else:
                md[c] = "" if pd.isna(v) else v
        # Chroma metadata 值建議是 str/int/float/bool
        if "row_index" in md:
            try:
                md["row_index"] = int(md["row_index"])
            except Exception:
                md["row_index"] = -1
        metadatas.append(md)

    print(f"準備建庫：{len(ids)} 筆")

    # 先算 embeddings
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # ⚠️ 如果之前已經 add 過，可能會撞 id
    # 這裡採用「先刪同 id 再 add」的簡單策略（適合你現在的規模）
    try:
        col.delete(ids=ids)
    except Exception:
        pass

    col.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print("✅ 建庫完成！")
    print("Chroma 目錄：", CHROMA_DIR)
    print("Collection：", COLLECTION_NAME)
    print("目前總數：", col.count())


if __name__ == "__main__":
    main()
