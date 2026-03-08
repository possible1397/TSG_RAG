import os
import re
import pandas as pd

# =========================
# 全域 Debug 參數
# =========================
DEBUG_COLLAPSE_STATS = True   # 想看合併統計就 True，不想看就 False
TOPK_GROUPS = 10              # 印出合併列數最多的前幾名群組


# =========================
# 小工具：確保欄位存在
# =========================
def ensure_col(df: pd.DataFrame, col: str, default: str = "") -> None:
    """確保 df 一定有 col 欄位，沒有就建立；並轉成字串且 NaN -> ''。"""
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna("").astype(str)


def is_trivial_text(s: str) -> bool:
    """判斷是否是垃圾文字（空白、nan、只有符號如 ':'）"""
    t = (s or "").strip()
    if t == "" or t.lower() == "nan":
        return True
    if t in [":", "：", "-", "—"]:
        return True
    if re.fullmatch(r"[:：\-\—\s]+", t):
        return True
    return False


def looks_like_path(t: str) -> bool:
    """判斷是否像路徑/連結，避免 symptom 抓到這些"""
    t = (t or "").strip()
    return (
        t.startswith("\\\\")
        or re.match(r"^[A-Za-z]:\\", t) is not None
        or t.startswith("http://")
        or t.startswith("https://")
    )


def normalize_colname(s: str) -> str:
    s = str(s)
    s = s.replace("\n", "").replace("\r", "")
    s = s.replace("　", " ")  # 全形空白 -> 半形
    s = re.sub(r"\s+", " ", s).strip()  # 多空白合一
    return s


# =========================
# key / text normalize 小工具（collapse 與 debug 共享）
# =========================
def _norm_spaces(s: str) -> str:
    s = ("" if s is None else str(s))
    s = s.replace("\r", "\n")
    s = s.replace("　", " ")
    s = re.sub(r"\n+", "\n", s)     # 多換行合一
    s = re.sub(r"[ \t]+", " ", s)   # 多空白合一
    return s.strip()


def _first_token_by_newline(s: str) -> str:
    """若 cell 裡有多行（多機台/多日期），保守取第一段，避免多事件被合併成 1 筆。"""
    s = _norm_spaces(s)
    if "\n" in s:
        return s.split("\n", 1)[0].strip()
    return s


def _normalize_event_keys_for_debug(_df: pd.DataFrame) -> pd.DataFrame:
    _df = _df.copy()
    for c in ["Client", "Machine_ID", "Date", "Supplier"]:
        if c not in _df.columns:
            _df[c] = ""
        _df[c] = _df[c].fillna("").astype(str)

    _df["Client"] = _df["Client"].map(_norm_spaces).str.replace("\n", " ", regex=False).str.strip()
    _df["Machine_ID"] = _df["Machine_ID"].map(_first_token_by_newline)
    _df["Date"] = _df["Date"].map(_first_token_by_newline)
    _df["Supplier"] = _df["Supplier"].map(_norm_spaces).str.replace("\n", " ", regex=False).str.strip()
    return _df


# =========================
# 多列事件合併（不出 FutureWarning）
# =========================
def collapse_multiline_events(df: pd.DataFrame, key_cols, text_cols, hint_col="__hint"):
    """
    把同一事件的多行內容合併成一行：
    - 先用 key_cols 分群
    - 針對每個群，從 text_cols 組合出 hint（取前 N 字，並去空白/換行）
    - 再用 key_cols + hint 分群做 agg 合併
    """

    # ---- 0) 防呆：確保欄位存在 ----
    df = df.copy()
    for c in key_cols + text_cols:
        if c not in df.columns:
            df[c] = ""

    # ---- 1) 清理：把 NaN 變空字串 ----
    for c in key_cols + text_cols:
        df[c] = df[c].fillna("").astype(str)

    # ---- 2) hint：用 transform 取每個群第一個「像樣的」文字作為 hint ----
    def make_hint_series(s: pd.Series, n=30) -> pd.Series:
        cleaned = s.fillna("").astype(str).map(lambda x: x.replace("\r", " ").replace("\n", " ").strip())
        first_non_empty = ""
        for v in cleaned.tolist():
            if v and v.lower() != "nan":
                first_non_empty = v
                break
        hint = first_non_empty[:n]
        return pd.Series([hint] * len(s), index=s.index)

    base_text_col = text_cols[0]  # 以 Description 當 hint 最穩
    df[hint_col] = df.groupby(key_cols, dropna=False)[base_text_col].transform(make_hint_series)

    # ---- 3) 第二輪：用 key + hint 分群做 agg 合併 ----
    def join_non_empty(values):
        vals = []
        for v in values:
            v = str(v).strip()
            if v and v.lower() != "nan":
                vals.append(v)
        # 去重但保序
        seen = set()
        out = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return "\n".join(out).strip()

    agg_dict = {}
    for c in key_cols:
        agg_dict[c] = "first"
    for c in text_cols:
        agg_dict[c] = join_non_empty

    out = (
        df.groupby(key_cols + [hint_col], dropna=False, sort=False)
          .agg(agg_dict)
          .reset_index(drop=True)
    )
    return out


# =========================
# Date 解析（v4）
# =========================
def parse_date_column(df: pd.DataFrame, colname: str = "Date") -> pd.DataFrame:
    if colname not in df.columns:
        return df

    col = df[colname]
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    if pd.api.types.is_datetime64_any_dtype(col):
        dt = col
    else:
        num = pd.to_numeric(col, errors="coerce")
        is_excel_serial = num.notna() & (num > 20000)

        if is_excel_serial.any():
            dt.loc[is_excel_serial] = pd.to_datetime(
                num.loc[is_excel_serial],
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )

        rest = ~is_excel_serial
        if rest.any():
            s = col.loc[rest].astype(str).str.strip()
            s = s.str.replace(r"\s+\d{1,2}:\d{2}(:\d{2})?$", "", regex=True)
            s = s.str.replace(r"[./]", "-", regex=True)

            m = s.str.extract(r"^(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})$")
            ok = m["y"].notna()

            fixed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
            if ok.any():
                fixed.loc[ok] = pd.to_datetime(
                    m.loc[ok, "y"]
                    + "-"
                    + m.loc[ok, "m"].astype(int).astype(str).str.zfill(2)
                    + "-"
                    + m.loc[ok, "d"].astype(int).astype(str).str.zfill(2),
                    format="%Y-%m-%d",
                    errors="coerce",
                )
            dt.loc[rest] = fixed

    raw_str = col.astype(str).str.strip()
    raw_not_empty = col.notna() & (raw_str != "") & (raw_str.str.lower() != "nan")
    bad_mask = raw_not_empty & dt.isna()
    if bad_mask.any():
        bad_df = df.loc[bad_mask].copy()
        bad_df["RawDate"] = col.loc[bad_mask]
        keep_cols = [c for c in ["Client", "Machine_ID", "Description", "Result"] if c in bad_df.columns]
        bad_df = bad_df[["RawDate"] + keep_cols]
        bad_df.to_csv("bad_date_rows.csv", index=False, encoding="utf-8-sig")

    df[colname] = dt.dt.strftime("%Y-%m-%d")
    df.loc[dt.isna(), colname] = "UNKNOWN"
    return df


# =========================
# 清洗 trivial rows（並 dump 被刪列）
# =========================
def drop_trivial_rows(df: pd.DataFrame) -> pd.DataFrame:
    ensure_col(df, "Description")
    ensure_col(df, "Hardware")
    ensure_col(df, "Result")

    before_cnt = len(df)

    def norm_text(x) -> str:
        s = ("" if x is None else str(x)).strip()
        if s.lower() in ["nan", "none", "null"]:
            return ""
        return s

    d = df["Description"].map(norm_text)
    r = df["Result"].map(norm_text)
    h = df["Hardware"].map(norm_text)
    joined = (d + " " + r + " " + h).str.strip()

    all_empty = (joined == "")
    only_symbols = joined.str.fullmatch(r"[:：\-\—\s]+").fillna(False)
    too_short = joined.str.len() < 3
    drop_mask = all_empty | (only_symbols & too_short)

    if drop_mask.any():
        dropped_df = df.loc[drop_mask, ["Client", "Machine_ID", "Date", "Description", "Result", "Hardware"]].copy()
        dropped_df.to_csv("dropped_trivial_rows.csv", index=False, encoding="utf-8-sig")

    df2 = df.loc[~drop_mask].copy()
    after_cnt = len(df2)

    dropped = before_cnt - after_cnt
    drop_empty = int(all_empty.sum())
    drop_symbols = int((only_symbols & too_short).sum())
    print(f"[Clean] drop trivial rows: {before_cnt} -> {after_cnt} (dropped {dropped}) | empty={drop_empty}, symbols={drop_symbols}")

    return df2


# =========================
# 特徵抽取（RAG 欄位）
# =========================
def extract_features_row(row: pd.Series) -> pd.Series:
    desc = (row.get("Description", "") or "").strip()
    hw = (row.get("Hardware", "") or "").strip()
    res = (row.get("Result", "") or "").strip()
    full_text_search = f"{desc} {hw} {res}".strip()

    category = "其他/未分類"
    if re.search(r"(?i)(Trigger|訊號|雜訊|干擾|接觸不良)", full_text_search):
        category = "訊號與干擾"
    elif re.search(r"(?i)(光源|閃爍|亮度|控制器|Strobe|2D站|Relay)", full_text_search):
        category = "光源系統"
    elif re.search(r"(?i)(取像失敗|Count 0|黑條|掃像失敗|全黑|影像晃動|影像錯位|影像異常)", full_text_search):
        category = "取像異常"
    elif re.search(r"(?i)(OVP|PCIe|接線|燒毀|插槽|斷線|接頭|TTL|跳電)", full_text_search):
        category = "硬體損壞/配線"
    elif re.search(r"(?i)(Driver|軟體|Win10|初始化|藍屏|UI|無法儲存|防毒|格式化|權限)", full_text_search):
        category = "軟體與驅動"

    keywords = [
        "OVP卡", "Trigger卡", "影像擷取卡", "五代卡", "Camera Link",
        "RS232", "光學尺", "PCIe插槽", "PCIe", "線材", "相機", "TTL卡",
        "光源控制器", "光控", "Relay", "離子槍", "CCD", "硬碟", "VLS", "防毒軟體"
    ]
    found = []
    for key in keywords:
        if re.search(re.escape(key), full_text_search, flags=re.IGNORECASE):
            found.append(key)
    key_component = ", ".join(sorted(set(found))) if found else "N/A"

    symptom_base = desc if not is_trivial_text(desc) else res
    symptom_text = re.sub(r"(經查修|經排查|客戶回報|現場確認|問題[:：])", "", symptom_base).strip()
    parts = [p.strip() for p in re.split(r"[，,。\n]", symptom_text) if p.strip()]

    symptom = "N/A"
    for p in parts:
        if not looks_like_path(p) and len(p) >= 2:
            symptom = p
            break
    if symptom == "N/A":
        symptom = (symptom_text[:30] + "...") if symptom_text else "N/A"

    if len(res) > 2 and not is_trivial_text(res):
        solution = res
    else:
        sentences = [s.strip() for s in re.split(r"[，,。\n]", desc) if s.strip()]
        solution = sentences[-1] if sentences else "需進一步確認"

    root_cause = "N/A"
    cause_match = re.search(r"(因為|由於|導致|原因為|原因[:：])([^，,。\n]+)", full_text_search)
    if cause_match:
        root_cause = cause_match.group(2).strip()
    elif "頻寬不足" in full_text_search:
        root_cause = "頻寬不足"
    elif "短路" in full_text_search:
        root_cause = "線路短路"

    full_content = (
        f"發生日期: {row.get('Date', '')}\n"
        f"客戶: {row.get('Client', '')}\n"
        f"機台編號: {row.get('Machine_ID', '')}\n"
        f"問題分類: {category}\n"
        f"關鍵元件: {key_component}\n"
        f"問題現象: {desc}\n"
        f"維修處理: {res}\n"
        f"更換硬體: {hw}"
    )

    return pd.Series([category, key_component, symptom, solution, root_cause, full_content])


# =========================
# 核心：清洗 + 合併 + 特徵抽取
# =========================
def clean_and_structure_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").copy()

    # 0) 必要欄位先補齊
    for c in ["Client", "Machine_ID", "Date", "Supplier", "Description", "Hardware", "Result"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)

    # ✅ 1) Forward fill（修正版）
    # 你前面 ensure_col 已經把 NaN 變 ""，ffill 會失效
    # 所以這裡要先把空字串轉成 NA 再 ffill
    fill_cols = ["Client", "Machine_ID", "Date", "Supplier"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .replace(r"^\s*$", pd.NA, regex=True)
                .ffill()
                .fillna("")
            )

    # 2) collapse 前統計（debug）
    if DEBUG_COLLAPSE_STATS:
        df_dbg = _normalize_event_keys_for_debug(df)
        key_cols_dbg = ["Client", "Machine_ID", "Date", "Supplier"]
        before_rows = len(df_dbg)
        grp_sizes = df_dbg.groupby(key_cols_dbg, dropna=False).size().sort_values(ascending=False)

        print(f"[Collapse] before rows={before_rows}, unique_events={len(grp_sizes)}")
        top = grp_sizes.head(TOPK_GROUPS)
        if len(top) > 0:
            print("[Collapse] top merged events (rows per event):")
            for idx, sz in top.items():
                if int(sz) <= 1:
                    break
                client, mid, date, sup = idx if isinstance(idx, tuple) else ("", "", str(idx), "")
                print(f"  - {int(sz)} rows | Client={client} | Machine_ID={mid} | Date={date} | Supplier={sup}")

    # ✅ 3) 合併多列事件（修正：要給 key_cols / text_cols）
    key_cols = ["Client", "Machine_ID", "Date", "Supplier"]
    text_cols = ["Description", "Hardware", "Result"]
    df = collapse_multiline_events(df, key_cols=key_cols, text_cols=text_cols)

    if DEBUG_COLLAPSE_STATS:
        after_rows = len(df)
        print(f"[Collapse] after rows={after_rows}, reduced_by={before_rows - after_rows}")

    # 4) Date parse
    df = parse_date_column(df, "Date")

    # 5) drop trivial rows（並 dump）
    df = drop_trivial_rows(df)

    # 6) 特徵抽取
    new_cols = ["Problem_Category", "Key_Component", "Symptom", "Solution", "Root_Cause", "Full_Content"]
    df[new_cols] = df.apply(extract_features_row, axis=1)

    final_cols = [
        "Date", "Client", "Machine_ID", "Supplier",
        "Description", "Result", "Hardware",  # ✅保留原始欄位
        "Problem_Category", "Key_Component", "Symptom", "Solution", "Root_Cause", "Full_Content"
    ]
    return df[[c for c in final_cols if c in df.columns]].copy()


# =========================
# 三種版型：自動判斷 mapping
# =========================
SCHEMA_A = {
    "客戶": "Client",
    "機台編號": "Machine_ID",
    "出機日期": "Ship_Date",
    "發生日期": "Date",
    "問題": "Issue",
    "說明": "Description",
    "備註": "Remark",
}

SCHEMA_B = {
    "客戶": "Client",
    "機台編號": "Machine_ID",
    "發生日期": "Date",
    "供應商": "Supplier",
    "硬體": "Hardware",
    "說明": "Description",
    "備註": "Remark",
    "處理結果": "Result",
}

SCHEMA_C = {
    "客戶": "Client",
    "機台編號": "Machine_ID",
    "出機日期": "Ship_Date",
    "發生日期": "Date",
    "硬體損壞": "Hardware",
    "說明": "Description",
    "處理結果": "Result",
}


def pick_schema_mapping(df: pd.DataFrame):
    cols = set([str(c).strip() for c in df.columns])

    if "供應商" in cols or ("硬體" in cols and "處理結果" in cols):
        return "B", SCHEMA_B

    if "硬體損壞" in cols:
        return "C", SCHEMA_C

    if "問題" in cols or "備註" in cols:
        return "A", SCHEMA_A

    return None, None


def read_excel_all_sheets(file_path: str):
    sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    out = {}
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = [normalize_colname(c) for c in df.columns]
        df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", na=False)]
        out[name] = df
    return out


# =========================
# 你要處理的檔案
# =========================
file_paths = [
    "C:/Users/Terry/PycharmProjects/TSG_RAG/20251227-LineScan Strip機台軟硬體問題記錄.xlsx",
]


def series_or_blank(df: pd.DataFrame, col: str) -> pd.Series:
    """永遠回傳 Series；欄位不存在就回傳同長度空字串 Series。"""
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def main():
    dfs_to_concat = []
    print("開始處理檔案（多 Sheet 模式）...")

    for file_path in file_paths:
        if os.path.basename(file_path).startswith("~$"):
            print(f"⚠️ 暫存檔跳過: {file_path}")
            continue

        if not os.path.exists(file_path):
            print(f"⚠️ 找不到檔案: {file_path}，跳過。")
            continue

        print(f"\n處理檔案: {file_path}")

        sheets = read_excel_all_sheets(file_path)
        if not sheets:
            print("⚠️ 這個檔案沒有可讀的 sheet，跳過。")
            continue

        for sheet_name, df in sheets.items():
            try:
                if df is None or df.empty:
                    continue

                schema_name, mapping = pick_schema_mapping(df)
                print(f"    columns: {list(df.columns)}")
                if mapping is None:
                    print(f"  - 跳過 sheet: {sheet_name}（欄位不符合 A/B/C）")
                    continue

                print(f"  - 解析 sheet: {sheet_name} schema={schema_name} rows={len(df)}")

                # 1) 欄位更名（中文 -> 英文統一欄名）
                df = df.rename(columns=mapping).copy()

                # 2) schema 特殊處理（先補齊欄位）
                for c in ["Client", "Machine_ID", "Date", "Supplier", "Hardware", "Description", "Result", "Remark"]:
                    ensure_col(df, c)

                if schema_name == "A":
                    ensure_col(df, "Issue")
                    df["Description"] = (df["Issue"].astype(str) + " " + df["Description"].astype(str)).str.strip()
                    df["Result"] = (df["Result"].astype(str) + " " + df["Remark"].astype(str)).str.strip()

                elif schema_name == "B":
                    df["Result"] = (df["Result"].astype(str) + " " + df["Remark"].astype(str)).str.strip()

                elif schema_name == "C":
                    pass

                # 3) 清洗 + 特徵抽取
                df_clean = clean_and_structure_log(df)

                # 4) 來源標籤
                df_clean["Source_File"] = os.path.basename(file_path)
                df_clean["Source_Sheet"] = sheet_name
                df_clean["Schema"] = schema_name

                # 5) RowIndex（追溯用）
                df_clean = df_clean.reset_index(drop=True)
                df_clean["RowIndex"] = df_clean.index.astype(int)

                # 6) Doc_ID
                machine_s = series_or_blank(df_clean, "Machine_ID")
                date_s = series_or_blank(df_clean, "Date")

                df_clean["Doc_ID"] = (
                    df_clean["Source_File"].astype(str)
                    + "|"
                    + df_clean["Source_Sheet"].astype(str)
                    + "|"
                    + df_clean["RowIndex"].astype(str)
                    + "|"
                    + machine_s
                    + "|"
                    + date_s
                )

                dfs_to_concat.append(df_clean)

            except Exception as e:
                print(f"❌ sheet={sheet_name} 發生錯誤: {e}")
                continue

    if dfs_to_concat:
        final_df = pd.concat(dfs_to_concat, ignore_index=True)

        output_filename = "Combined_Maintenance_Log_Cleaned.csv"
        final_df.to_csv(output_filename, index=False, encoding="utf-8-sig")

        print(f"\n✅ 處理完成！")
        print(f"總共合併了 {len(final_df)} 筆維修記錄。")
        print(f"檔案已儲存為: {output_filename}")

        show_cols = [c for c in ["Date", "Problem_Category", "Symptom"] if c in final_df.columns]
        if show_cols:
            print(final_df[show_cols].head())
        else:
            print(final_df.head())

    else:
        print("沒有任何檔案被處理。")


if __name__ == "__main__":
    main()
