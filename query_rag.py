import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db_local"
COLLECTION_NAME = "maintenance_logs"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

def main():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"找不到 {CHROMA_DIR}，請先跑 build_chroma_db_local.py")

    print("載入 embedding 模型：", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    col = client.get_collection(name=COLLECTION_NAME)

    while True:
        q = input("\n請輸入你的問題（輸入 q 離開）：").strip()
        if q.lower() in ["q", "quit", "exit"]:
            break

        # 轉 query embedding
        q_emb = model.encode([q], normalize_embeddings=True).tolist()

        # 你也可以加 where filter（可選）
        # where = {"client": "Client_A"}   # 只查某客戶
        where = None

        res = col.query(
            query_embeddings=q_emb,
            n_results=TOP_K,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        mds = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        print("\n=== Top Matches ===")
        for i, (doc_id, doc, md, dist) in enumerate(zip(ids, docs, mds, dists), start=1):
            print(f"\n[{i}] doc_id={doc_id}  distance={dist:.4f}")
            print(f"    date={md.get('date','')}, client={md.get('client','')}, machine={md.get('machine_id','')}")
            print(f"    category={md.get('problem_category','')}, component={md.get('key_component','')}")
            print("---- content preview ----")
            print(doc[:600])  # 只印前 600 字，避免太長

if __name__ == "__main__":
    main()
