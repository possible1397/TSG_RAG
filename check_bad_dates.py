import pandas as pd
import os
import re

# 跟你 Dataclean.py 一樣的設定
file_configs = [
    {
        "path": "C:/Users/Terry/PycharmProjects/TSG_RAG/20251227-LineScan Strip機台軟硬體問題記錄_3.xlsx",
        "mapping": {
            '客戶': 'Client', '機台編號': 'Machine_ID', '發生日期': 'Date',
            '硬體損壞': 'Hardware', '說明': 'Description', '處理結果': 'Result'
        }
    },
    {
        "path": "C:/Users/Terry/PycharmProjects/TSG_RAG/20251227-LineScan Strip機台軟硬體問題記錄_2.xlsx",
        "mapping": {
            '客戶': 'Client', '機台編號': 'Machine_ID', '發生日期': 'Date', '供應商': 'Supplier',
            '硬體': 'Hardware', '說明': 'Description', '處理結果': 'Result'
        }
    },
    {
        "path": "C:/Users/Terry/PycharmProjects/TSG_RAG/20251227-LineScan Strip機台軟硬體問題記錄_1.xlsx",
        "mapping": {
            '客戶': 'Client', '機台編號': 'Machine_ID', '發生日期': 'Date',
            '說明': 'Description_Detail', '問題': 'Description_Title', '備註': 'Result'
        },
        "is_software": True
    }
]


def normalize_date_no_guess(series: pd.Series) -> pd.Series:
    """
    用「不猜格式」的方式抓出日期問題：
    - datetime: 直接用
    - Excel 序號: 用 origin 轉
    - 字串: 只接受 YYYY-MM-DD（允許 / . 會先換成 -）
    """
    col = series

    if pd.api.types.is_datetime64_any_dtype(col):
        return col

    num = pd.to_numeric(col, errors='coerce')
    is_excel_serial = num.notna() & (num > 20000)

    dt = pd.Series(pd.NaT, index=col.index, dtype='datetime64[ns]')

    if is_excel_serial.any():
        dt.loc[is_excel_serial] = pd.to_datetime(
            num.loc[is_excel_serial],
            unit='D',
            origin='1899-12-30',
            errors='coerce'
        )

    rest = ~is_excel_serial
    if rest.any():
        s = col.loc[rest].astype(str).str.strip()
        s = s.str.replace(r"[./]", "-", regex=True)
        # 只接受這個格式，不做 dateutil 猜測
        dt.loc[rest] = pd.to_datetime(s, errors='coerce', format='%Y-%m-%d')

    return dt


def main():
    bad_rows = []

    for cfg in file_configs:
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"⚠️ 找不到檔案：{path}")
            continue

        print(f"\n檢查檔案：{path}")

        xls = pd.ExcelFile(path, engine="openpyxl")
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            df.columns = df.columns.map(str)
            df = df.loc[:, ~df.columns.str.contains(r'^Unnamed', na=False)]

            # 欄位更名
            df = df.rename(columns=cfg["mapping"])

            # 軟體檔合併 Description（只是為了報表可讀）
            if cfg.get("is_software"):
                title = df.get('Description_Title', '').fillna('')
                detail = df.get('Description_Detail', '').fillna('')
                df['Description'] = title.astype(str) + ": " + detail.astype(str)

            if "Date" not in df.columns:
                print(f"  - Sheet {sheet}: ❌ 找不到 Date 欄位（mapping 可能沒對到）")
                continue

            raw_date = df["Date"]
            parsed = normalize_date_no_guess(raw_date)

            # 有問題：原始不空，但 parse 失敗 -> NaT
            raw_str = raw_date.astype(str).str.strip()
            raw_not_empty = (
                    raw_date.notna()
                    & (raw_str != "")
                    & (raw_str.str.lower() != "nan")
            )

            bad_mask = raw_not_empty & parsed.isna()

            bad_count = int(bad_mask.sum())
            print(f"  - Sheet {sheet}: 找到 {bad_count} 筆日期解析失敗")

            if bad_count == 0:
                continue

            # 為了讓你回 Excel 對照：加上「Excel列號」
            # pandas index 0 對應到 Excel 的資料列，通常要 +2（1:表頭那列，2:第一筆資料）
            # 但如果你的 Excel 表頭不在第 1 列，就會有偏差；至少可以當參考。
            excel_row = bad_mask[bad_mask].index.to_series().apply(lambda i: int(i) + 2)

            preview_cols = []
            for c in ["Client", "Machine_ID", "Description", "Result"]:
                if c in df.columns:
                    preview_cols.append(c)

            for idx in bad_mask[bad_mask].index:
                bad_rows.append({
                    "File": os.path.basename(path),
                    "FullPath": path,
                    "Sheet": sheet,
                    "ExcelRow(approx)": int(idx) + 2,
                    "RawDate": raw_date.loc[idx],
                    **{c: df.loc[idx, c] for c in preview_cols}
                })

    if bad_rows:
        out = pd.DataFrame(bad_rows)
        out.to_csv("bad_dates_report.csv", index=False, encoding="utf-8-sig")
        print("\n✅ 已輸出 bad_dates_report.csv（請用 Excel 打開對照）")
        print(out.head(10))
    else:
        print("\n✅ 沒有找到日期解析失敗的列（或日期都能被規則解析）")


if __name__ == "__main__":
    main()
