import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from rag_engine import RAGEngine


class ChatUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSG RAG SOP - 售服排查助手")
        self.geometry("1100x700")
        self.minsize(980, 620)

        self.engine = None
        self._build_layout()
        self._init_engine()

    def _build_layout(self):
        # 主容器：左右分割（左：聊天，右：案例）
        self.columnconfigure(0, weight=7)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        # Left: chat
        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        # Chat history (Text + Scroll)
        chat_frame = ttk.Frame(left)
        chat_frame.grid(row=0, column=0, sticky="nsew")
        chat_frame.rowconfigure(0, weight=1)
        chat_frame.columnconfigure(0, weight=1)

        self.chat_text = tk.Text(chat_frame, wrap="word", font=("Consolas", 11))
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        self.chat_text.configure(state="disabled")

        chat_scroll = ttk.Scrollbar(chat_frame, command=self.chat_text.yview)
        chat_scroll.grid(row=0, column=1, sticky="ns")
        self.chat_text["yscrollcommand"] = chat_scroll.set

        # input area
        input_frame = ttk.Frame(left)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        input_frame.columnconfigure(0, weight=1)

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_var, font=("Microsoft JhengHei", 11))
        self.input_entry.grid(row=0, column=0, sticky="ew")
        self.input_entry.bind("<Return>", lambda e: self.on_send())

        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=0, column=1, padx=(8, 0))

        self.send_btn = ttk.Button(btn_frame, text="送出", command=self.on_send)
        self.send_btn.grid(row=0, column=0, padx=(0, 6))

        self.clear_btn = ttk.Button(btn_frame, text="清除對話", command=self.on_clear)
        self.clear_btn.grid(row=0, column=1)

        # Right: matches list + detail
        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        title = ttk.Label(right, text="相似案例 (TopK)", font=("Microsoft JhengHei", 12, "bold"))
        title.grid(row=0, column=0, sticky="w")

        self.match_list = tk.Listbox(right, height=10)
        self.match_list.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        self.match_list.bind("<<ListboxSelect>>", self.on_select_match)

        self.detail_text = tk.Text(right, wrap="word", font=("Consolas", 10))
        self.detail_text.grid(row=2, column=0, sticky="nsew")
        self.detail_text.configure(state="disabled")

        detail_scroll = ttk.Scrollbar(right, command=self.detail_text.yview)
        detail_scroll.grid(row=2, column=1, sticky="ns")
        self.detail_text["yscrollcommand"] = detail_scroll.set

        self.current_matches = []

    def _init_engine(self):
        try:
            self._append_chat("系統", "初始化中：載入 embedding 模型與 Chroma 資料庫...\n（第一次可能較久）")
            self.update_idletasks()
            self.engine = RAGEngine(top_k=6)
            self._append_chat("系統", "✅ 初始化完成！你可以開始輸入問題，例如：Count 0 怎麼排查？")
            self.input_entry.focus_set()
        except Exception as e:
            messagebox.showerror("初始化失敗", f"無法初始化 RAGEngine：\n{e}")
            self.destroy()

    def _append_chat(self, role: str, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", f"[{ts}] {role}：\n{message}\n\n")
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _set_detail(self, message: str):
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", message)
        self.detail_text.configure(state="disabled")

    def on_clear(self):
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", "end")
        self.chat_text.configure(state="disabled")
        self.match_list.delete(0, "end")
        self._set_detail("")
        self.current_matches = []
        self._append_chat("系統", "對話已清除。")

    def on_send(self):
        q = self.input_var.get().strip()
        if not q:
            return
        self.input_var.set("")
        self._append_chat("你", q)

        try:
            result = self.engine.ask(q)

            # 主回答（Checklist + 統計）
            msg = []
            msg.append("========================================================================")
            msg.append("🧰 售服現場排查 SOP（Checklist 型 + RAG 輔助）")
            msg.append("========================================================================")
            msg.append(f"問題：{result['question']}")
            msg.append(f"SOP 類型判斷：{result['sop_type']}")
            msg.append("")
            msg.append(result["sop_text"])
            msg.append("")
            msg.append("------------------------------------------------------------------------")
            msg.append("📌 RAG 相似案例彙整（TopK 統計）")
            msg.append("------------------------------------------------------------------------")
            stats = result["stats"]
            msg.append(f"常見關聯元件： {stats.get('components','N/A')}")
            msg.append(f"常見問題分類： {stats.get('categories','N/A')}")
            msg.append(f"客戶分布（參考）： {stats.get('clients','N/A')}")

            self._append_chat("助手", "\n".join(msg))

            # 右側 Top Matches
            self.current_matches = result["matches"]
            self.match_list.delete(0, "end")
            for i, m in enumerate(self.current_matches, start=1):
                line = f"[{i}] {m.get('date','')} | {m.get('client','')} | {m.get('machine_id','')} | {m.get('problem_category','')}"
                self.match_list.insert("end", line)

            self._set_detail("點右側案例可查看 doc_id 與完整摘要")

        except Exception as e:
            self._append_chat("助手", f"⚠️ 查詢失敗：{e}")

    def on_select_match(self, event):
        if not self.current_matches:
            return
        sel = self.match_list.curselection()
        if not sel:
            return
        idx = sel[0]
        m = self.current_matches[idx]

        detail = []
        detail.append(f"doc_id: {m.get('doc_id','')}")
        detail.append(f"distance: {m.get('distance', None)}")
        detail.append(f"date: {m.get('date','')}")
        detail.append(f"client: {m.get('client','')}")
        detail.append(f"machine_id: {m.get('machine_id','')}")
        detail.append(f"problem_category: {m.get('problem_category','')}")
        detail.append(f"key_component: {m.get('key_component','')}")
        detail.append("")
        detail.append("preview:")
        detail.append(m.get("preview",""))
        detail.append("")
        detail.append("content (partial):")
        detail.append((m.get("content","") or "")[:2000])

        self._set_detail("\n".join(detail))


if __name__ == "__main__":
    app = ChatUI()
    app.mainloop()
