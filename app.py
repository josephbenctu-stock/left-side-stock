"""
左側低估值選股 Pro App
------------------------------------------------------------
用途：
1. 使用官方 OpenAPI 的本益比、股價淨值比、殖利率與行情資料，先做估值濾網。
2. 盡量合併官方月營收資料，避免低本益比但營收轉壞的價值陷阱。
3. 用 yfinance 下載歷史 K 線，計算左側交易常用條件：
   - 60 / 120 日高點回檔幅度
   - RSI 低檔回升
   - KD 低檔黃金交叉
   - 接近 60 日低點、120MA、240MA 支撐
   - 20 日均量
   - 近期不再創低
4. 自動產生：型態分類、價值陷阱警示、分批進場觀察價、停損價、壓力價。
5. 新增財報品質濾網、大盤風險燈號、產業強弱濾網、部位風險計算器。
6. 提供自選股 / 觀察名單狀態追蹤 / 單檔簡易技術回測。

注意：
這是研究與選股輔助工具，不是投資建議。資料來源可能延遲、缺漏或格式變動，
實際交易前請再與券商、交易所、櫃買中心或公開資訊觀測站資料交叉確認。
"""

from __future__ import annotations

import io
import math
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# ========= 基本設定 =========

st.set_page_config(
    page_title="左側低估值選股器 Pro Max",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

TWSE_PE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TWSE_QUOTE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_REVENUE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap15_L"

TPEX_PE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"
TPEX_QUOTE_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
TPEX_REVENUE_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap39_O"
MARKET_INDEX_TICKER = "^TWII"
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_API_URL_V3 = "https://api.finmindtrade.com/api/v3/data"


REQUEST_HEADERS = {
    # 用比較像一般瀏覽器的 UA，降低雲端環境被官方站誤判的機率。
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json,text/csv,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

# TWSE OpenAPI 在 Streamlit Cloud 或部分雲端環境偶爾會回傳空白 / HTML，
# 造成 json() 出現 Expecting value。這裡加入 TWSE 備援下載網址。
URL_FALLBACKS = {
    TWSE_PE_URL: [
        TWSE_PE_URL,
        "https://www.twse.com.tw/exchangeReport/BWIBBU_ALL?response=open_data",
        "https://www.twse.com.tw/rwd/zh/exchangeReport/BWIBBU_ALL?response=open_data",
    ],
    TWSE_QUOTE_URL: [
        TWSE_QUOTE_URL,
        "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data",
        "https://www.twse.com.tw/rwd/zh/exchangeReport/STOCK_DAY_ALL?response=open_data",
    ],
    TWSE_REVENUE_URL: [
        TWSE_REVENUE_URL,
        "https://www.twse.com.tw/rwd/zh/opendata/t187ap15_L?response=open_data",
        "https://mopsfin.twse.com.tw/opendata/t187ap15_L.csv",
    ],
}


# ========= 工具函式 =========


def clean_code(x: object) -> str:
    """把股票代號整理成純字串。"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace("=", "").replace('"', "").replace("'", "")
    out = ""
    for ch in s:
        if ch.isdigit():
            out += ch
        elif out:
            break
    return out


def to_number(x: object) -> float:
    """把 OpenAPI 可能出現的 --、逗號、百分比、空值轉成 float。"""
    if x is None or pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"", "--", "-", "NaN", "nan", "N/A", "不適用", "無", "None"}:
        return np.nan
    s = s.replace(",", "").replace("%", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def sane_percent(x: object) -> float:
    """清理百分比。公開資訊中若出現極端代碼值，轉成空值。"""
    v = to_number(x)
    if pd.isna(v):
        return np.nan
    # 有些資料會用 999999.99 表示無法計算。
    if abs(v) > 9999:
        return np.nan
    return float(v)


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """在資料表中找可能的欄位名稱。"""
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    normalized_map = {str(c).lower().replace(" ", "").replace("_", ""): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
        key2 = cand.lower().replace(" ", "").replace("_", "")
        if key2 in normalized_map:
            return normalized_map[key2]
    return None


def _records_from_json_or_csv(text: str, url: str) -> list:
    """把 API 回傳內容轉成 list[dict]；可處理 JSON 與 open_data CSV。"""
    if not text or not text.strip():
        raise ValueError("官方 API 回傳空內容")

    stripped = text.lstrip("\ufeff\n\r\t ")

    # HTML 通常代表官方站暫時阻擋、維護或回傳錯誤頁。
    if stripped.startswith("<"):
        preview = stripped[:80].replace("\n", " ").replace("\r", " ")
        raise ValueError(f"官方 API 回傳非資料頁面：{preview}")

    # 先嘗試 JSON。
    try:
        import json

        data = json.loads(stripped)
        if isinstance(data, dict):
            for key in ["data", "result", "tables", "items"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 再嘗試 CSV / open_data。
    try:
        df = pd.read_csv(io.StringIO(text))
        # 有些 open_data CSV 會多出空欄或說明列，先做基本清理。
        df = df.dropna(how="all")
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        if not df.empty and len(df.columns) >= 2:
            return df.to_dict("records")
    except Exception:
        pass

    preview = stripped[:120].replace("\n", " ").replace("\r", " ")
    raise ValueError(f"無法辨識的 API 回傳格式：{url}；內容前段：{preview}")


def request_json(url: str, timeout: int = 25) -> list:
    """下載資料。優先走官方 OpenAPI，失敗時自動嘗試 TWSE 備援 open_data。"""
    urls = URL_FALLBACKS.get(url, [url])
    last_error: Optional[Exception] = None

    for candidate in urls:
        try:
            resp = requests.get(candidate, headers=REQUEST_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return _records_from_json_or_csv(resp.text, candidate)
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)

    raise ValueError(f"全部資料源皆下載失敗，最後錯誤：{last_error}")



# ========= FinMind 自動資料源 =========


def _date_str(days_ago: int = 0) -> str:
    return (datetime.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _safe_json(resp: requests.Response, label: str) -> dict:
    """讀取 API JSON；若回傳 HTML 或空白，改成可讀錯誤訊息。"""
    text = resp.text or ""
    stripped = text.lstrip("\ufeff\n\r\t ")
    if not stripped:
        raise ValueError(f"{label} 回傳空內容")
    if stripped.startswith("<"):
        preview = stripped[:100].replace("\n", " ").replace("\r", " ")
        raise ValueError(f"{label} 回傳 HTML/非 JSON：{preview}")
    try:
        return resp.json()
    except Exception as exc:
        preview = stripped[:160].replace("\n", " ").replace("\r", " ")
        raise ValueError(f"{label} JSON 解析失敗：{exc}；內容前段：{preview}")


def finmind_request(dataset: str, token: str = "", start_date: str = "", end_date: str = "", data_id: str = "") -> list:
    """FinMind 資料下載。優先使用 v4；失敗時嘗試 v3 參數。"""
    token = (token or "").strip()
    params_v4 = {"dataset": dataset}
    if data_id:
        params_v4["data_id"] = data_id
    if start_date:
        params_v4["start_date"] = start_date
    if end_date:
        params_v4["end_date"] = end_date
    if token:
        params_v4["token"] = token

    last_error: Optional[Exception] = None
    try:
        resp = requests.get(FINMIND_API_URL, params=params_v4, headers=REQUEST_HEADERS, timeout=35)
        resp.raise_for_status()
        data = _safe_json(resp, f"FinMind {dataset}")
        rows = data.get("data", data if isinstance(data, list) else [])
        if isinstance(rows, list) and len(rows) > 0:
            return rows
        # FinMind 有時候成功但 data 為空，也讓 v3 嘗試一次。
        last_error = ValueError(f"FinMind v4 {dataset} 無資料")
    except Exception as exc:
        last_error = exc

    # v3 fallback。部分舊文件使用 stock_id/date 參數。
    params_v3 = {"dataset": dataset}
    if data_id:
        params_v3["stock_id"] = data_id
    if start_date:
        params_v3["date"] = start_date
    if token:
        params_v3["token"] = token
    try:
        resp = requests.get(FINMIND_API_URL_V3, params=params_v3, headers=REQUEST_HEADERS, timeout=35)
        resp.raise_for_status()
        data = _safe_json(resp, f"FinMind v3 {dataset}")
        rows = data.get("data", data if isinstance(data, list) else [])
        if isinstance(rows, list):
            return rows
    except Exception as exc:
        last_error = exc

    raise ValueError(f"FinMind {dataset} 下載失敗：{last_error}")


def map_finmind_market(x: object) -> str:
    s = str(x).strip().lower()
    if any(k in s for k in ["上市", "twse", "tse", "sii", "listed", "exchange"]):
        return "上市"
    if any(k in s for k in ["上櫃", "tpex", "otc", "gre tai", "gtsm"]):
        return "上櫃"
    return ""


def normalize_finmind_info(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=["代號", "名稱", "產業別", "市場"])
    code_col = find_col(df, ["stock_id", "股票代號", "代號", "code"])
    name_col = find_col(df, ["stock_name", "股票名稱", "公司名稱", "名稱", "name"])
    industry_col = find_col(df, ["industry_category", "industry", "產業別", "產業類別"])
    market_col = find_col(df, ["type", "market", "exchange", "市場", "上市櫃", "listing"])
    out = pd.DataFrame()
    out["代號"] = df[code_col].map(clean_code) if code_col else ""
    out["名稱"] = df[name_col].astype(str).str.strip() if name_col else ""
    out["產業別"] = df[industry_col].astype(str).str.strip() if industry_col else ""
    out["市場"] = df[market_col].map(map_finmind_market) if market_col else ""
    out = out[out["代號"].str.len().between(4, 6)]
    # 僅保留一般股票代號，排除 ETF/權證雜訊可由 type/名稱輔助；缺欄時至少保留代號格式。
    if "名稱" in out.columns:
        out = out[~out["名稱"].astype(str).str.contains("購|售|牛|熊|權證|認購|認售", na=False)]
    return out.drop_duplicates(subset=["代號"], keep="last")


def normalize_finmind_per(raw: list, info_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame()
    code_col = find_col(df, ["stock_id", "股票代號", "代號", "code"])
    date_col = find_col(df, ["date", "日期", "資料日期"])
    pe_col = find_col(df, ["PER", "pe", "本益比", "PEratio"])
    pb_col = find_col(df, ["PBR", "pb", "股價淨值比", "PBratio"])
    div_col = find_col(df, ["dividend_yield", "DividendYield", "殖利率", "殖利率%"])
    out = pd.DataFrame()
    out["代號"] = df[code_col].map(clean_code) if code_col else ""
    out["估值資料日期"] = df[date_col].astype(str) if date_col else ""
    out["本益比"] = df[pe_col].map(to_number) if pe_col else np.nan
    out["股價淨值比"] = df[pb_col].map(to_number) if pb_col else np.nan
    out["殖利率%"] = df[div_col].map(to_number) if div_col else np.nan
    out = out[out["代號"].str.len().between(4, 6)]
    if "估值資料日期" in out.columns:
        out = out.sort_values(["代號", "估值資料日期"], ascending=[True, False])
    out = out.drop_duplicates(subset=["代號"], keep="first")
    if info_df is not None and not info_df.empty:
        out = out.merge(info_df[["代號", "名稱", "產業別", "市場"]], on="代號", how="left")
    else:
        out["名稱"] = ""
        out["產業別"] = ""
        out["市場"] = ""
    out = out[out["市場"].isin(["上市", "上櫃"])]
    out["yfinance代號"] = out["代號"] + np.where(out["市場"].eq("上市"), ".TW", ".TWO")
    out["資料源"] = "FinMind"
    return out


def normalize_finmind_price(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=["代號", "收盤價", "今日成交量張"])
    code_col = find_col(df, ["stock_id", "股票代號", "代號", "code"])
    date_col = find_col(df, ["date", "日期"])
    close_col = find_col(df, ["close", "Close", "收盤價", "收盤"])
    vol_col = find_col(df, ["Trading_Volume", "Trading_Volume", "TradingVolume", "volume", "成交股數", "成交量"])
    out = pd.DataFrame()
    out["代號"] = df[code_col].map(clean_code) if code_col else ""
    out["行情日期"] = df[date_col].astype(str) if date_col else ""
    out["收盤價"] = df[close_col].map(to_number) if close_col else np.nan
    vol = df[vol_col].map(to_number) if vol_col else np.nan
    # FinMind TaiwanStockPrice 的 Trading_Volume 通常為股數；若數值已是張數，除以1000會偏小，但仍只作輔助排序。
    out["今日成交量張"] = vol / 1000 if vol_col else np.nan
    out = out[out["代號"].str.len().between(4, 6)]
    if "行情日期" in out.columns:
        out = out.sort_values(["代號", "行情日期"], ascending=[True, False])
    return out.drop_duplicates(subset=["代號"], keep="first")


def normalize_finmind_revenue(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=["代號"])
    code_col = find_col(df, ["stock_id", "股票代號", "公司代號", "代號", "code"])
    date_col = find_col(df, ["date", "資料年月", "營收年月", "出表日期"])
    rev_col = find_col(df, ["revenue", "月營收", "本月營收", "當月營收", "營業收入-當月營收"])
    last_year_col = find_col(df, ["last_year_revenue", "去年同月營收", "去年當月營收", "營業收入-去年當月營收"])
    yoy_col = find_col(df, ["revenue_growth_rate", "YoY", "yoy", "月營收YoY%", "去年同月增減(%)", "營業收入-去年同月增減(%)"])
    acc_col = find_col(df, ["cumulative_revenue", "累計營收", "當月累計營收", "累計營業收入-當月累計營收"])
    acc_yoy_col = find_col(df, ["cumulative_revenue_growth_rate", "累計營收YoY%", "累計營收年增率", "累計營業收入-前期比較增減(%)"])
    year_col = find_col(df, ["revenue_year", "資料年度", "年度", "year"])
    month_col = find_col(df, ["revenue_month", "資料月份", "月份", "month"])
    out = pd.DataFrame()
    out["代號"] = df[code_col].map(clean_code) if code_col else ""
    out["營收資料日期"] = df[date_col].astype(str) if date_col else ""
    out["營收年度"] = df[year_col].map(to_number) if year_col else np.nan
    out["營收月份"] = df[month_col].map(to_number) if month_col else np.nan
    out["月營收"] = df[rev_col].map(to_number) if rev_col else np.nan
    out["去年同月營收"] = df[last_year_col].map(to_number) if last_year_col else np.nan
    out["月營收YoY%"] = df[yoy_col].map(sane_percent) if yoy_col else np.nan
    out["累計營收"] = df[acc_col].map(to_number) if acc_col else np.nan
    out["累計營收YoY%"] = df[acc_yoy_col].map(sane_percent) if acc_yoy_col else np.nan
    need_yoy = out["月營收YoY%"].isna() & out["月營收"].notna() & out["去年同月營收"].notna() & (out["去年同月營收"] != 0)
    out.loc[need_yoy, "月營收YoY%"] = (out.loc[need_yoy, "月營收"] / out.loc[need_yoy, "去年同月營收"] - 1) * 100
    out = out[out["代號"].str.len().between(4, 6)]
    if "營收資料日期" in out.columns:
        out = out.sort_values(["代號", "營收資料日期"], ascending=[True, False])
    elif out[["營收年度", "營收月份"]].notna().any().any():
        out = out.sort_values(["代號", "營收年度", "營收月份"], ascending=[True, False, False])
    return out.drop_duplicates(subset=["代號"], keep="first")



# ========= Yahoo 批次估值備援 =========

YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"


def chunked(seq: List[str], size: int = 80) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _yahoo_percent(val: object) -> float:
    """Yahoo 有時用 0.03 表示 3%，有時用 3 表示 3%，統一轉成百分比。"""
    v = to_number(val)
    if pd.isna(v):
        return np.nan
    if abs(v) <= 1:
        return v * 100
    return v


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_yahoo_valuation_fallback(info_records: Tuple[Tuple[str, str, str, str], ...], include_twse: bool, include_tpex: bool) -> pd.DataFrame:
    """當 FinMind / TWSE 無法提供全市場估值時，用 Yahoo Finance quote 批次補 PE/PB/殖利率。

    info_records: (代號, 名稱, 產業別, 市場)
    這不是交易所官方資料，但可避免 Streamlit Cloud 被 TWSE 擋住時整個 App 無資料。
    """
    rows = []
    info = pd.DataFrame(info_records, columns=["代號", "名稱", "產業別", "市場"])
    if info.empty:
        return pd.DataFrame()
    wanted = []
    if include_twse:
        wanted.append("上市")
    if include_tpex:
        wanted.append("上櫃")
    info = info[info["市場"].isin(wanted)].copy()
    info = info[info["代號"].astype(str).str.len().between(4, 6)]
    info = info.drop_duplicates(subset=["代號", "市場"], keep="last")
    if info.empty:
        return pd.DataFrame()

    symbol_to_meta = {}
    symbols = []
    for _, r in info.iterrows():
        suffix = ".TW" if r["市場"] == "上市" else ".TWO"
        sym = f"{r['代號']}{suffix}"
        symbols.append(sym)
        symbol_to_meta[sym] = r

    for part in chunked(symbols, 80):
        try:
            resp = requests.get(
                YAHOO_QUOTE_URL,
                params={"symbols": ",".join(part), "lang": "zh-TW", "region": "TW"},
                headers=REQUEST_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = _safe_json(resp, "Yahoo quote")
            results = data.get("quoteResponse", {}).get("result", []) if isinstance(data, dict) else []
        except Exception:
            results = []
        for item in results:
            sym = item.get("symbol", "")
            meta = symbol_to_meta.get(sym)
            if meta is None:
                continue
            pe = first_notna(item.get("trailingPE"), item.get("forwardPE"))
            pb = first_notna(item.get("priceToBook"), item.get("bookValue"))
            # bookValue 不是 P/B，若沒有 priceToBook，不用 bookValue 代替。
            pb = item.get("priceToBook", np.nan)
            dy = first_notna(item.get("trailingAnnualDividendYield"), item.get("dividendYield"))
            rows.append({
                "代號": meta["代號"],
                "名稱": first_notna(item.get("shortName"), item.get("longName"), meta["名稱"]),
                "產業別": meta["產業別"],
                "市場": meta["市場"],
                "估值資料日期": _date_str(0),
                "本益比": to_number(pe),
                "股價淨值比": to_number(pb),
                "殖利率%": _yahoo_percent(dy),
                "收盤價": to_number(first_notna(item.get("regularMarketPrice"), item.get("postMarketPrice"), item.get("preMarketPrice"))),
                "今日成交量張": to_number(item.get("regularMarketVolume")) / 1000,
                "yfinance代號": sym,
                "資料源": "Yahoo備援",
            })
        time.sleep(0.1)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame()
    out = out.dropna(subset=["本益比"])
    out = out[out["本益比"] > 0]
    for col in ["營收年度", "營收月份", "月營收", "月營收MoM%", "月營收YoY%", "累計營收YoY%"]:
        if col not in out.columns:
            out[col] = np.nan
    return compute_industry_relative_columns(out).reset_index(drop=True)

def compute_industry_relative_columns(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return base
    if "產業別" in base.columns and base["產業別"].astype(str).str.len().gt(0).any():
        base["產業平均本益比"] = base.groupby(["市場", "產業別"])["本益比"].transform(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().mean())
        base["產業平均P/B"] = base.groupby(["市場", "產業別"])["股價淨值比"].transform(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().mean())
        base["相對產業本益比%"] = base["本益比"] / base["產業平均本益比"] * 100
        base["相對產業P/B%"] = base["股價淨值比"] / base["產業平均P/B"] * 100
    else:
        base["產業平均本益比"] = np.nan
        base["產業平均P/B"] = np.nan
        base["相對產業本益比%"] = np.nan
        base["相對產業P/B%"] = np.nan
    return base


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_finmind_data(include_twse: bool, include_tpex: bool, include_revenue: bool, token: str = "") -> pd.DataFrame:
    """用 FinMind 作為主要資料源，避免 TWSE 在 Streamlit Cloud 回傳 HTML 的問題。"""
    errors = []
    end = _date_str(0)
    per_start = _date_str(45)
    price_start = _date_str(14)
    revenue_start = _date_str(430)

    try:
        info_raw = finmind_request("TaiwanStockInfo", token=token)
        info_df = normalize_finmind_info(info_raw)
    except Exception as exc:
        info_df = pd.DataFrame()
        errors.append(f"FinMind 股票基本資料下載失敗：{exc}")

    try:
        per_raw = finmind_request("TaiwanStockPER", token=token, start_date=per_start, end_date=end)
        base = normalize_finmind_per(per_raw, info_df)
    except Exception as exc:
        errors.append(f"FinMind 本益比/PBR/殖利率資料下載失敗：{exc}")
        base = pd.DataFrame()

    # TaiwanStockPER 在部分情況下不能全市場直接下載，或會要求個股 stock_id。
    # 若 FinMind 全市場估值失敗，改用 Yahoo Finance 批次 quote 作估值備援，避免 App 完全沒有資料。
    if base.empty and info_df is not None and not info_df.empty:
        try:
            info_records = tuple(info_df[["代號", "名稱", "產業別", "市場"]].fillna("").astype(str).itertuples(index=False, name=None))
            base = load_yahoo_valuation_fallback(info_records, include_twse, include_tpex)
            if not base.empty:
                errors.append("已改用 Yahoo 批次估值備援；本益比/PBR/殖利率非交易所官方資料，請作為篩選輔助。")
        except Exception as exc:
            errors.append(f"Yahoo 批次估值備援失敗：{exc}")
            base = pd.DataFrame()

    if base.empty:
        for msg in errors:
            st.warning(msg)
        return pd.DataFrame()

    wanted_markets = []
    if include_twse:
        wanted_markets.append("上市")
    if include_tpex:
        wanted_markets.append("上櫃")
    base = base[base["市場"].isin(wanted_markets)].copy()

    try:
        price_raw = finmind_request("TaiwanStockPrice", token=token, start_date=price_start, end_date=end)
        price_df = normalize_finmind_price(price_raw)
        if not price_df.empty:
            base = base.merge(price_df, on="代號", how="left")
    except Exception as exc:
        errors.append(f"FinMind 日行情資料下載失敗：{exc}")

    if "收盤價" not in base.columns:
        base["收盤價"] = np.nan
    if "今日成交量張" not in base.columns:
        base["今日成交量張"] = np.nan

    if include_revenue:
        try:
            revenue_raw = finmind_request("TaiwanStockMonthRevenue", token=token, start_date=revenue_start, end_date=end)
            revenue_df = normalize_finmind_revenue(revenue_raw)
            if not revenue_df.empty:
                base = base.merge(revenue_df, on="代號", how="left")
        except Exception as exc:
            errors.append(f"FinMind 月營收資料下載失敗：{exc}")

    for col in ["產業別", "營收年度", "營收月份", "月營收", "月營收MoM%", "月營收YoY%", "累計營收YoY%"]:
        if col not in base.columns:
            base[col] = np.nan if col != "產業別" else ""

    base = base.dropna(subset=["本益比"])
    base = base[base["本益比"] > 0]
    base = compute_industry_relative_columns(base)

    for msg in errors:
        st.warning(msg)
    return base.reset_index(drop=True)


def first_notna(*vals: object) -> object:
    for val in vals:
        if val is not None and not pd.isna(val):
            return val
    return np.nan


def normalize_pe_table(raw: list, market: str) -> pd.DataFrame:
    """統一上市 / 上櫃本益比表欄位。"""
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame()

    code_col = find_col(df, ["Code", "股票代號", "代號", "SecuritiesCompanyCode", "SecuritiesCode"])
    name_col = find_col(df, ["Name", "名稱", "公司名稱", "SecuritiesCompanyName"])
    pe_col = find_col(df, ["PEratio", "本益比", "P/E ratio", "P/E", "PER", "PriceEarningRatio"])
    pb_col = find_col(df, ["PBratio", "股價淨值比", "P/B ratio", "P/B", "PBR", "PriceBookRatio"])
    div_col = find_col(df, ["DividendYield", "殖利率", "殖利率(%)", "Yield", "Dividend yield"])
    date_col = find_col(df, ["Date", "資料日期", "日期"])

    out = pd.DataFrame()
    out["代號"] = df[code_col].map(clean_code) if code_col else ""
    out["名稱"] = df[name_col].astype(str).str.strip() if name_col else ""
    out["本益比"] = df[pe_col].map(to_number) if pe_col else np.nan
    out["股價淨值比"] = df[pb_col].map(to_number) if pb_col else np.nan
    out["殖利率%"] = df[div_col].map(to_number) if div_col else np.nan
    out["估值資料日期"] = df[date_col].astype(str).str.strip() if date_col else ""
    out["市場"] = market
    out["yfinance代號"] = out["代號"] + (".TW" if market == "上市" else ".TWO")
    out = out[out["代號"].str.len().between(4, 6)]
    out = out.drop_duplicates(subset=["代號", "市場"])
    return out


def normalize_quote_table(raw: list, market: str) -> pd.DataFrame:
    """統一今日收盤 / 成交量欄位，主要作為預先排序與輔助顯示。"""
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=["代號", "市場", "收盤價", "今日成交量張"])

    code_col = find_col(df, ["Code", "股票代號", "代號", "SecuritiesCompanyCode", "SecuritiesCode"])
    close_col = find_col(df, ["ClosingPrice", "收盤", "收盤價", "Close", "LatestPrice"])
    volume_col = find_col(df, ["TradeVolume", "成交股數", "成交量", "Volume"])

    out = pd.DataFrame()
    out["代號"] = df[code_col].map(clean_code) if code_col else ""
    out["市場"] = market
    out["收盤價"] = df[close_col].map(to_number) if close_col else np.nan
    out["今日成交量張"] = df[volume_col].map(to_number) / 1000 if volume_col else np.nan
    out = out[out["代號"].str.len().between(4, 6)]
    return out.drop_duplicates(subset=["代號", "市場"])


def normalize_revenue_table(raw: list, market: str) -> pd.DataFrame:
    """統一月營收資料欄位。不同官方 OpenAPI 欄名可能略有差異，所以用多組候選欄位。"""
    df = pd.DataFrame(raw)
    if df.empty:
        return pd.DataFrame(columns=["代號", "市場"])

    code_col = find_col(df, ["公司代號", "出表日期", "Code", "SecuritiesCompanyCode", "股票代號", "代號"])
    # 上面候選含「出表日期」是為了防格式錯誤；下面會用 clean_code + 長度過濾避免誤用。
    if code_col == "出表日期":
        maybe = find_col(df, ["公司代號", "股票代號", "代號", "Code", "SecuritiesCompanyCode"])
        code_col = maybe or code_col

    name_col = find_col(df, ["公司名稱", "名稱", "Name", "SecuritiesCompanyName"])
    industry_col = find_col(df, ["產業別", "Industry", "industry"])
    year_col = find_col(df, ["資料年度", "年度", "Year", "year"])
    month_col = find_col(df, ["資料月份", "月份", "Month", "month"])

    current_rev_col = find_col(df, [
        "營業收入-當月營收", "當月營收", "本月營收", "本月營收淨額", "CurrentMonthRevenue",
        "Revenue Current Month", "CurrentRevenue",
    ])
    prev_month_rev_col = find_col(df, [
        "營業收入-上月營收", "上月營收", "上月營收淨額", "LastMonthRevenue", "PreviousMonthRevenue",
    ])
    last_year_rev_col = find_col(df, [
        "營業收入-去年當月營收", "去年當月營收", "去年同月營收", "去年本月營收淨額", "LastYearMonthRevenue",
    ])
    mom_col = find_col(df, [
        "營業收入-上月比較增減(%)", "上月比較增減(%)", "增減變動比例（%）", "MoM", "mom",
    ])
    yoy_col = find_col(df, [
        "營業收入-去年同月增減(%)", "去年同月增減(%)", "年增率", "YoY", "yoy",
    ])
    acc_rev_col = find_col(df, [
        "累計營業收入-當月累計營收", "當月累計營收", "累計營收", "CumulativeRevenue",
    ])
    acc_yoy_col = find_col(df, [
        "累計營業收入-前期比較增減(%)", "前期比較增減(%)", "累計營收年增率", "CumulativeYoY",
    ])

    out = pd.DataFrame()
    out["代號"] = df[code_col].map(clean_code) if code_col else ""
    out["名稱_營收"] = df[name_col].astype(str).str.strip() if name_col else ""
    out["產業別"] = df[industry_col].astype(str).str.strip() if industry_col else ""
    out["營收年度"] = df[year_col].map(to_number) if year_col else np.nan
    out["營收月份"] = df[month_col].map(to_number) if month_col else np.nan
    out["月營收"] = df[current_rev_col].map(to_number) if current_rev_col else np.nan
    out["上月營收"] = df[prev_month_rev_col].map(to_number) if prev_month_rev_col else np.nan
    out["去年同月營收"] = df[last_year_rev_col].map(to_number) if last_year_rev_col else np.nan
    out["月營收MoM%"] = df[mom_col].map(sane_percent) if mom_col else np.nan
    out["月營收YoY%"] = df[yoy_col].map(sane_percent) if yoy_col else np.nan
    out["累計營收"] = df[acc_rev_col].map(to_number) if acc_rev_col else np.nan
    out["累計營收YoY%"] = df[acc_yoy_col].map(sane_percent) if acc_yoy_col else np.nan
    out["市場"] = market

    # 若沒有官方 YoY 欄位，但有本月與去年同月，就自行計算。
    need_yoy = out["月營收YoY%"].isna() & out["月營收"].notna() & out["去年同月營收"].notna() & (out["去年同月營收"] != 0)
    out.loc[need_yoy, "月營收YoY%"] = (out.loc[need_yoy, "月營收"] / out.loc[need_yoy, "去年同月營收"] - 1) * 100

    need_mom = out["月營收MoM%"].isna() & out["月營收"].notna() & out["上月營收"].notna() & (out["上月營收"] != 0)
    out.loc[need_mom, "月營收MoM%"] = (out.loc[need_mom, "月營收"] / out.loc[need_mom, "上月營收"] - 1) * 100

    out = out[out["代號"].str.len().between(4, 6)]
    # 若資料源含多月份，保留同股票最新年度月份。
    if out[["營收年度", "營收月份"]].notna().any().any():
        out = out.sort_values(["代號", "營收年度", "營收月份"], ascending=[True, False, False])
    out = out.drop_duplicates(subset=["代號", "市場"], keep="first")
    return out.reset_index(drop=True)


def load_uploaded_records(uploaded_file, label: str) -> list:
    """讀取使用者上傳的 TWSE JSON / CSV 備援資料。"""
    if uploaded_file is None:
        return []
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="replace")
    return _records_from_json_or_csv(text, label)


def assemble_market_table(pe_df: pd.DataFrame, quote_df: pd.DataFrame, revenue_df: pd.DataFrame) -> pd.DataFrame:
    """把估值、行情、月營收三份表合併成主資料表。"""
    if pe_df is None or pe_df.empty:
        return pd.DataFrame()

    base = pe_df.copy()
    if quote_df is not None and not quote_df.empty:
        base = base.merge(quote_df, on=["代號", "市場"], how="left")
    else:
        if "收盤價" not in base.columns:
            base["收盤價"] = np.nan
        if "今日成交量張" not in base.columns:
            base["今日成交量張"] = np.nan

    if revenue_df is not None and not revenue_df.empty:
        base = base.merge(revenue_df, on=["代號", "市場"], how="left")
        if "名稱_營收" in base.columns:
            base["名稱"] = base["名稱"].where(base["名稱"].astype(str).str.len() > 0, base.get("名稱_營收", ""))
    else:
        for col in ["產業別", "月營收YoY%", "累計營收YoY%"]:
            if col not in base.columns:
                base[col] = np.nan if col.endswith("%") else ""

    return base


def build_uploaded_twse_data(pe_upload, quote_upload, revenue_upload, include_revenue: bool) -> pd.DataFrame:
    """建立使用者上傳的上市備援資料表。至少需要上市估值資料。"""
    try:
        pe_raw = load_uploaded_records(pe_upload, "上傳上市估值資料")
        if not pe_raw:
            return pd.DataFrame()
        pe_df = normalize_pe_table(pe_raw, "上市")

        quote_df = pd.DataFrame()
        if quote_upload is not None:
            quote_raw = load_uploaded_records(quote_upload, "上傳上市行情資料")
            quote_df = normalize_quote_table(quote_raw, "上市")

        revenue_df = pd.DataFrame()
        if include_revenue and revenue_upload is not None:
            revenue_raw = load_uploaded_records(revenue_upload, "上傳上市月營收資料")
            revenue_df = normalize_revenue_table(revenue_raw, "上市")

        return assemble_market_table(pe_df, quote_df, revenue_df)
    except Exception as exc:
        st.warning(f"上市備援上傳資料讀取失敗：{exc}")
        return pd.DataFrame()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_official_data(include_twse: bool, include_tpex: bool, include_revenue: bool) -> pd.DataFrame:
    """下載官方估值、行情、月營收資料。快取 30 分鐘。"""
    pe_frames = []
    quote_frames = []
    revenue_frames = []
    errors = []

    if include_twse:
        try:
            pe_frames.append(normalize_pe_table(request_json(TWSE_PE_URL), "上市"))
        except Exception as exc:
            errors.append(f"上市估值資料下載失敗：{exc}")
        try:
            quote_frames.append(normalize_quote_table(request_json(TWSE_QUOTE_URL), "上市"))
        except Exception as exc:
            errors.append(f"上市行情資料下載失敗：{exc}")
        if include_revenue:
            try:
                revenue_frames.append(normalize_revenue_table(request_json(TWSE_REVENUE_URL), "上市"))
            except Exception as exc:
                errors.append(f"上市月營收資料下載失敗：{exc}")

    if include_tpex:
        try:
            pe_frames.append(normalize_pe_table(request_json(TPEX_PE_URL), "上櫃"))
        except Exception as exc:
            errors.append(f"上櫃估值資料下載失敗：{exc}")
        try:
            quote_frames.append(normalize_quote_table(request_json(TPEX_QUOTE_URL), "上櫃"))
        except Exception as exc:
            errors.append(f"上櫃行情資料下載失敗：{exc}")
        if include_revenue:
            try:
                revenue_frames.append(normalize_revenue_table(request_json(TPEX_REVENUE_URL), "上櫃"))
            except Exception as exc:
                errors.append(f"上櫃月營收資料下載失敗：{exc}")

    if errors:
        for msg in errors:
            st.warning(msg)

    if not pe_frames:
        return pd.DataFrame()

    base = pd.concat(pe_frames, ignore_index=True)
    quotes = pd.concat(quote_frames, ignore_index=True) if quote_frames else pd.DataFrame()
    revenues = pd.concat(revenue_frames, ignore_index=True) if revenue_frames else pd.DataFrame()

    if not quotes.empty:
        base = base.merge(quotes, on=["代號", "市場"], how="left")
    else:
        base["收盤價"] = np.nan
        base["今日成交量張"] = np.nan

    if not revenues.empty:
        base = base.merge(revenues, on=["代號", "市場"], how="left")
        base["名稱"] = base["名稱"].where(base["名稱"].astype(str).str.len() > 0, base.get("名稱_營收", ""))
    else:
        for col in ["產業別", "營收年度", "營收月份", "月營收", "月營收MoM%", "月營收YoY%", "累計營收YoY%"]:
            base[col] = np.nan if col != "產業別" else ""

    base = base.dropna(subset=["本益比"])
    base = base[base["本益比"] > 0]

    # 產業相對估值。若產業別不足，欄位保留為空值。
    if "產業別" in base.columns and base["產業別"].astype(str).str.len().gt(0).any():
        base["產業平均本益比"] = base.groupby(["市場", "產業別"])["本益比"].transform(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().mean())
        base["產業平均P/B"] = base.groupby(["市場", "產業別"])["股價淨值比"].transform(lambda s: s.replace([np.inf, -np.inf], np.nan).dropna().mean())
        base["相對產業本益比%"] = base["本益比"] / base["產業平均本益比"] * 100
        base["相對產業P/B%"] = base["股價淨值比"] / base["產業平均P/B"] * 100
    else:
        base["產業平均本益比"] = np.nan
        base["產業平均P/B"] = np.nan
        base["相對產業本益比%"] = np.nan
        base["相對產業P/B%"] = np.nan

    return base.reset_index(drop=True)


# ========= 財報品質 / 大盤 / 產業與風控工具 =========


def normalize_financial_quality_upload(file_obj) -> pd.DataFrame:
    """讀取使用者自行上傳的財報品質 CSV。

    建議欄位可使用中文或英文：
    代號 / code、近四季EPS / eps_ttm、EPS年增% / eps_growth、毛利率% / gross_margin、
    營益率% / operating_margin、ROE% / roe、營業現金流 / operating_cashflow、負債比% / debt_ratio。
    """
    if file_obj is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding="big5")
    if df.empty:
        return pd.DataFrame()

    code_col = find_col(df, ["代號", "股票代號", "公司代號", "code", "ticker", "symbol"])
    name_col = find_col(df, ["名稱", "公司名稱", "name"])
    eps_col = find_col(df, ["近四季EPS", "EPS", "eps_ttm", "trailing_eps", "每股盈餘"])
    eps_growth_col = find_col(df, ["EPS年增%", "EPS成長%", "獲利成長率%", "eps_growth", "earnings_growth"])
    gross_col = find_col(df, ["毛利率%", "毛利率", "gross_margin", "grossMargins"])
    op_col = find_col(df, ["營益率%", "營業利益率%", "營業利益率", "operating_margin", "operatingMargins"])
    roe_col = find_col(df, ["ROE%", "ROE", "return_on_equity", "returnOnEquity"])
    ocf_col = find_col(df, ["營業現金流", "營運現金流", "operating_cashflow", "operatingCashflow"])
    debt_col = find_col(df, ["負債比%", "負債比", "debt_ratio", "debtToEquity", "負債權益比"])

    out = pd.DataFrame()
    out["代號"] = df[code_col].map(clean_code) if code_col else ""
    out["名稱_財報"] = df[name_col].astype(str).str.strip() if name_col else ""
    out["近四季EPS"] = df[eps_col].map(to_number) if eps_col else np.nan
    out["EPS年增%"] = df[eps_growth_col].map(sane_percent) if eps_growth_col else np.nan
    out["毛利率%"] = df[gross_col].map(sane_percent) if gross_col else np.nan
    out["營業利益率%"] = df[op_col].map(sane_percent) if op_col else np.nan
    out["ROE%"] = df[roe_col].map(sane_percent) if roe_col else np.nan
    out["營業現金流"] = df[ocf_col].map(to_number) if ocf_col else np.nan
    out["負債比%"] = df[debt_col].map(sane_percent) if debt_col else np.nan
    out["財報來源"] = "CSV"
    out = out[out["代號"].str.len().between(4, 6)]
    return out.drop_duplicates(subset=["代號"], keep="last")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_yahoo_financial_quality(tickers: Tuple[str, ...]) -> pd.DataFrame:
    """用 Yahoo Finance 補財報品質欄位。這個資料源不是官方資料，缺漏時不視為錯誤。"""
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        code = clean_code(ticker)
        if not code:
            continue
        def pct(key: str) -> float:
            val = info.get(key, np.nan)
            val = to_number(val)
            if pd.isna(val):
                return np.nan
            # Yahoo 的 margin / ROE 常用 0.123 表示 12.3%。debtToEquity 則常是 123.4。
            if abs(val) <= 2 and key != "debtToEquity":
                return val * 100
            return val
        rows.append({
            "代號": code,
            "近四季EPS": to_number(info.get("trailingEps")),
            "EPS年增%": pct("earningsQuarterlyGrowth"),
            "毛利率%": pct("grossMargins"),
            "營業利益率%": pct("operatingMargins"),
            "ROE%": pct("returnOnEquity"),
            "營業現金流": to_number(info.get("operatingCashflow")),
            "負債比%": pct("debtToEquity"),
            "財報來源": "Yahoo",
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["代號"], keep="last") if rows else pd.DataFrame()


def merge_financial_quality(base: pd.DataFrame, csv_df: pd.DataFrame, yahoo_df: pd.DataFrame) -> pd.DataFrame:
    """把 CSV 與 Yahoo 財報品質資料合併到官方估值資料。CSV 優先，Yahoo 補空。"""
    out = base.copy()
    fin_cols = ["近四季EPS", "EPS年增%", "毛利率%", "營業利益率%", "ROE%", "營業現金流", "負債比%", "財報來源"]
    for c in fin_cols:
        if c not in out.columns:
            out[c] = np.nan if c != "財報來源" else ""

    if yahoo_df is not None and not yahoo_df.empty:
        y = yahoo_df[[c for c in ["代號"] + fin_cols if c in yahoo_df.columns]].copy()
        out = out.merge(y, on="代號", how="left", suffixes=("", "_Yahoo"))
        for c in fin_cols:
            yc = c + "_Yahoo"
            if yc in out.columns:
                out[c] = out[c].where(out[c].notna() & (out[c].astype(str) != ""), out[yc])
                out = out.drop(columns=[yc])

    if csv_df is not None and not csv_df.empty:
        cdf = csv_df[[c for c in ["代號"] + fin_cols if c in csv_df.columns]].copy()
        out = out.merge(cdf, on="代號", how="left", suffixes=("", "_CSV"))
        for c in fin_cols:
            cc = c + "_CSV"
            if cc in out.columns:
                if c == "財報來源":
                    out[c] = out[cc].where(out[cc].astype(str).str.len() > 0, out[c])
                else:
                    out[c] = out[cc].where(out[cc].notna(), out[c])
                out = out.drop(columns=[cc])
    return out


def financial_quality_score_and_warning(row: pd.Series | Dict[str, object], params: ScreenParams) -> Tuple[int, str, bool]:
    """回傳：分數、警示、是否通過濾網。"""
    eps = to_number(row.get("近四季EPS"))
    eps_growth = sane_percent(row.get("EPS年增%"))
    gross = sane_percent(row.get("毛利率%"))
    op_margin = sane_percent(row.get("營業利益率%"))
    roe = sane_percent(row.get("ROE%"))
    ocf = to_number(row.get("營業現金流"))
    debt = sane_percent(row.get("負債比%"))
    values = [eps, eps_growth, gross, op_margin, roe, ocf, debt]
    has_any = any(not pd.isna(v) for v in values)

    if not params.use_fin_quality_filter:
        return 0, "", True
    if params.require_fin_quality_data and not has_any:
        return 0, "缺財報品質資料", False

    score = 0
    warnings = []
    ok = True

    if not pd.isna(eps):
        if eps >= params.eps_min:
            score += 4
        else:
            warnings.append("EPS偏低")
            ok = False
    elif params.require_fin_quality_data:
        warnings.append("缺EPS")
        ok = False

    if not pd.isna(eps_growth):
        if eps_growth >= params.eps_growth_min:
            score += 3
        else:
            warnings.append("獲利成長偏弱")
            ok = False

    if not pd.isna(gross):
        if gross >= params.gross_margin_min:
            score += 3
        else:
            warnings.append("毛利率偏低")
            ok = False

    if not pd.isna(op_margin):
        if op_margin >= params.operating_margin_min:
            score += 3
        else:
            warnings.append("營益率偏低")
            ok = False

    if not pd.isna(roe):
        if roe >= params.roe_min:
            score += 4
        else:
            warnings.append("ROE偏低")
            ok = False

    if params.require_positive_ocf:
        if not pd.isna(ocf):
            if ocf > 0:
                score += 2
            else:
                warnings.append("營業現金流非正")
                ok = False
        elif params.require_fin_quality_data:
            warnings.append("缺營業現金流")
            ok = False

    if not pd.isna(debt):
        if debt <= params.debt_ratio_max:
            score += 2
        else:
            warnings.append("負債比偏高")
            ok = False

    # 如果缺資料但不強制，給中性處理，不讓它直接淘汰。
    if not has_any and not params.require_fin_quality_data:
        return 0, "財報品質資料缺漏，僅作提示", True
    return int(min(score, 15)), "、".join(warnings), ok


def apply_financial_quality_filters(base: pd.DataFrame, params: ScreenParams) -> pd.DataFrame:
    if not params.use_fin_quality_filter or base.empty:
        return base
    rows = []
    for _, row in base.iterrows():
        fin_score, fin_warning, ok = financial_quality_score_and_warning(row, params)
        if ok:
            item = row.to_dict()
            item["財報品質分"] = fin_score
            item["財報品質警示"] = fin_warning
            rows.append(item)
    if not rows:
        return pd.DataFrame(columns=list(base.columns) + ["財報品質分", "財報品質警示"])
    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 20, show_spinner=False)
def load_market_risk(period: str = "12mo") -> Dict[str, object]:
    """下載加權指數並計算大盤風險燈號。"""
    try:
        raw = yf.download(MARKET_INDEX_TICKER, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    except Exception as exc:
        return {"燈號": "未知", "說明": f"大盤資料下載失敗：{exc}"}
    df = extract_one_symbol(raw, MARKET_INDEX_TICKER)
    if df.empty or len(df) < 80:
        return {"燈號": "未知", "說明": "大盤資料不足"}
    close = df["Close"].astype(float)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    rsi = compute_rsi(close, 14)
    ret20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else np.nan
    ret60 = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 61 else np.nan
    current = float(close.iloc[-1])
    above_ma20 = current >= float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else False
    above_ma60 = current >= float(ma60.iloc[-1]) if not pd.isna(ma60.iloc[-1]) else False
    above_ma120 = current >= float(ma120.iloc[-1]) if not pd.isna(ma120.iloc[-1]) else False
    rsi_now = float(rsi.iloc[-1])

    if above_ma20 and above_ma60 and rsi_now >= 50:
        light = "綠燈"
        note = "指數站上短中期均線，左側篩選可正常運作，但仍需分批。"
    elif above_ma60 or above_ma120 or rsi_now >= 45:
        light = "黃燈"
        note = "大盤仍在震盪，建議只做 A/C 類且降低部位。"
    else:
        light = "紅燈"
        note = "大盤偏弱，左側交易容易太早接刀，建議只觀察或大幅降低部位。"

    return {
        "燈號": light,
        "說明": note,
        "指數收盤": round(current, 2),
        "MA20": round(float(ma20.iloc[-1]), 2) if not pd.isna(ma20.iloc[-1]) else np.nan,
        "MA60": round(float(ma60.iloc[-1]), 2) if not pd.isna(ma60.iloc[-1]) else np.nan,
        "MA120": round(float(ma120.iloc[-1]), 2) if not pd.isna(ma120.iloc[-1]) else np.nan,
        "RSI14": round(rsi_now, 2),
        "20日報酬%": round(float(ret20), 2) if not pd.isna(ret20) else np.nan,
        "60日報酬%": round(float(ret60), 2) if not pd.isna(ret60) else np.nan,
    }


def add_industry_strength_columns(df: pd.DataFrame, market_ret20: float) -> pd.DataFrame:
    if df.empty or "產業別" not in df.columns:
        return df
    out = df.copy()
    out["產業別"] = out["產業別"].fillna("").replace("", "未分類")
    group_cols = ["市場", "產業別"] if "市場" in out.columns else ["產業別"]
    out["產業20日平均報酬%"] = out.groupby(group_cols)["20日報酬%"].transform("mean") if "20日報酬%" in out.columns else np.nan
    out["產業60日平均報酬%"] = out.groupby(group_cols)["60日報酬%"].transform("mean") if "60日報酬%" in out.columns else np.nan
    if "20日報酬%" in out.columns:
        out["產業20日上漲比例%"] = out.groupby(group_cols)["20日報酬%"].transform(lambda s: (s > 0).mean() * 100)
    else:
        out["產業20日上漲比例%"] = np.nan
    if not pd.isna(market_ret20):
        out["個股相對大盤20日%"] = out["20日報酬%"] - market_ret20
        out["產業相對大盤20日%"] = out["產業20日平均報酬%"] - market_ret20
    else:
        out["個股相對大盤20日%"] = np.nan
        out["產業相對大盤20日%"] = np.nan
    return out


def apply_industry_strength_filter(df: pd.DataFrame, params: ScreenParams) -> pd.DataFrame:
    if not params.use_industry_strength_filter or df.empty:
        return df
    cond = pd.Series(True, index=df.index)
    if "產業20日平均報酬%" in df.columns:
        cond &= df["產業20日平均報酬%"].isna() | (df["產業20日平均報酬%"] >= params.min_industry_ret20)
    if "產業相對大盤20日%" in df.columns:
        cond &= df["產業相對大盤20日%"].isna() | (df["產業相對大盤20日%"] >= params.min_industry_relative20)
    if "產業20日上漲比例%" in df.columns:
        cond &= df["產業20日上漲比例%"].isna() | (df["產業20日上漲比例%"] >= params.min_industry_breadth)
    return df[cond].reset_index(drop=True)


def determine_trade_status(row: pd.Series | Dict[str, object]) -> str:
    cur = to_number(row.get("收盤價_技術"))
    stop = to_number(row.get("停損價"))
    test_low = to_number(row.get("試單區間低"))
    test_high = to_number(row.get("試單區間高"))
    second = to_number(row.get("二次回測觀察價"))
    confirm = to_number(row.get("確認加碼價"))
    if pd.isna(cur):
        return "等待更新"
    if not pd.isna(stop) and cur <= stop:
        return "跌破停損"
    if not pd.isna(confirm) and cur >= confirm:
        return "站上確認加碼價"
    if not pd.isna(test_low) and not pd.isna(test_high) and test_low <= cur <= test_high:
        return "已到試單區"
    if not pd.isna(second) and abs(cur / second - 1) * 100 <= 2.0:
        return "二次回測中"
    if not pd.isna(test_low) and cur < test_low:
        return "低於試單區，等止跌"
    return "等待試單"


def add_position_plan_columns(df: pd.DataFrame, account_capital: float, risk_pct: float, max_position_pct: float,
                              tranche1_pct: float, tranche2_pct: float, tranche3_pct: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    risk_budget = account_capital * risk_pct / 100
    max_position_value = account_capital * max_position_pct / 100
    shares_list = []
    lots_list = []
    amount_list = []
    risk_per_share_list = []
    first_lots = []
    second_lots = []
    third_lots = []
    for _, row in out.iterrows():
        cur = to_number(row.get("收盤價_技術"))
        stop = to_number(row.get("停損價"))
        if pd.isna(cur) or pd.isna(stop) or cur <= stop or cur <= 0:
            shares = 0
            risk_per_share = np.nan
        else:
            risk_per_share = cur - stop
            by_risk = math.floor(risk_budget / risk_per_share)
            by_capital = math.floor(max_position_value / cur)
            shares = max(0, min(by_risk, by_capital))
            # 台股以零股也能交易，但策略規劃預設用整張呈現；不足一張仍保留股數。
            if shares >= 1000:
                shares = (shares // 1000) * 1000
        lots = shares / 1000 if shares else 0
        amount = shares * cur if shares else 0
        shares_list.append(int(shares))
        lots_list.append(round(lots, 2))
        amount_list.append(round(amount, 0))
        risk_per_share_list.append(round(float(risk_per_share), 2) if not pd.isna(risk_per_share) else np.nan)
        first_lots.append(round(lots * tranche1_pct / 100, 2))
        second_lots.append(round(lots * tranche2_pct / 100, 2))
        third_lots.append(round(lots * tranche3_pct / 100, 2))
    out["單股風險"] = risk_per_share_list
    out["建議總股數"] = shares_list
    out["建議總張數"] = lots_list
    out["建議部位金額"] = amount_list
    out["第一筆張數"] = first_lots
    out["第二筆張數"] = second_lots
    out["第三筆張數"] = third_lots
    return out


def update_watchlist_status(watchlist: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    if watchlist.empty or current_df.empty or "代號" not in watchlist.columns:
        return watchlist
    current_cols = [c for c in current_df.columns if c in {
        "代號", "市場", "名稱", "收盤價_技術", "左側總分", "左側型態", "風險警示", "交易狀態",
        "試單區間低", "試單區間高", "二次回測觀察價", "確認加碼價", "停損價", "第一壓力價", "第二壓力價",
        "RSI14", "60日回檔%", "產業20日平均報酬%", "產業相對大盤20日%",
    }]
    cur = current_df[current_cols].drop_duplicates(subset=["代號", "市場"], keep="last") if "市場" in current_cols else current_df[current_cols].drop_duplicates(subset=["代號"], keep="last")
    keys = ["代號", "市場"] if "市場" in watchlist.columns and "市場" in cur.columns else ["代號"]
    merged = watchlist.merge(cur, on=keys, how="left", suffixes=("", "_更新"))
    for c in current_cols:
        if c in keys:
            continue
        uc = c + "_更新"
        if uc in merged.columns:
            merged[c] = merged[uc].where(merged[uc].notna(), merged.get(c, np.nan))
            merged = merged.drop(columns=[uc])
    merged["狀態更新時間"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if "交易狀態" not in merged.columns:
        merged["交易狀態"] = merged.apply(determine_trade_status, axis=1)
    else:
        merged["交易狀態"] = merged.apply(determine_trade_status, axis=1)
    return merged


# ========= 技術分析 =========


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_kd(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(period).min()
    highest = high.rolling(period).max()
    rsv = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d


@dataclass
class ScreenParams:
    pe_max: float
    min_price: float
    use_pb_filter: bool
    pb_max: float
    use_dividend_filter: bool
    dividend_min: float
    use_revenue_filter: bool
    require_revenue_data: bool
    revenue_yoy_min: float
    revenue_acc_yoy_min: float
    use_fin_quality_filter: bool
    require_fin_quality_data: bool
    eps_min: float
    eps_growth_min: float
    gross_margin_min: float
    operating_margin_min: float
    roe_min: float
    debt_ratio_max: float
    require_positive_ocf: bool
    use_industry_pe_filter: bool
    industry_pe_ratio_max: float
    use_industry_strength_filter: bool
    min_industry_ret20: float
    min_industry_relative20: float
    min_industry_breadth: float
    min_avg_volume_lots: float
    dd60_min: float
    dd60_max: float
    rsi_max: float
    rsi_turn_days: int
    support_distance_max: float
    require_kd_cross: bool
    require_no_new_low: bool
    min_total_score: int
    stop_loss_buffer: float
    risk_reward_min: float
    pe_deep_value_level: float
    market_risk_mode: str


def extract_one_symbol(hist: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """從 yfinance 多股票下載結果中抽出單一股票。"""
    if hist.empty:
        return pd.DataFrame()

    if isinstance(hist.columns, pd.MultiIndex):
        if ticker in hist.columns.get_level_values(0):
            df = hist[ticker].copy()
        elif ticker in hist.columns.get_level_values(-1):
            df = hist.xs(ticker, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
    else:
        df = hist.copy()

    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna(how="any")
    return df[df["Close"] > 0]


@st.cache_data(ttl=60 * 20, show_spinner=False)
def download_history(tickers: List[str], period: str = "12mo") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )


def nearest_support(current: float, candidates: List[Tuple[str, float]]) -> Tuple[str, float, float]:
    valid = []
    for name, val in candidates:
        if val is not None and not pd.isna(val) and val > 0:
            valid.append((name, float(val), abs(current / float(val) - 1) * 100))
    if not valid:
        return "", np.nan, np.nan
    valid.sort(key=lambda x: x[2])
    return valid[0]


def classify_setup(row: Dict[str, object], params: ScreenParams) -> Tuple[str, str]:
    """依估值、營收、技術結構做左側型態分類與警示。"""
    pe = to_number(row.get("本益比"))
    pb = to_number(row.get("股價淨值比"))
    div_yield = to_number(row.get("殖利率%"))
    rev_yoy = sane_percent(row.get("月營收YoY%"))
    acc_yoy = sane_percent(row.get("累計營收YoY%"))
    dd60 = to_number(row.get("60日回檔%"))
    rsi = to_number(row.get("RSI14"))
    no_new_low = row.get("近期不破低") == "是"
    support_distance = to_number(row.get("支撐距離%"))
    vol_shrink = row.get("近5日量縮") == "是"

    warnings = []
    revenue_bad = False
    if not pd.isna(rev_yoy) and rev_yoy < params.revenue_yoy_min:
        revenue_bad = True
        warnings.append("月營收YoY偏弱")
    if not pd.isna(acc_yoy) and acc_yoy < params.revenue_acc_yoy_min:
        revenue_bad = True
        warnings.append("累計營收YoY偏弱")
    if not no_new_low:
        warnings.append("近期仍破低")
    if not pd.isna(pb) and params.use_pb_filter and pb > params.pb_max:
        warnings.append("P/B偏高")
    if not pd.isna(div_yield) and params.use_dividend_filter and div_yield < params.dividend_min:
        warnings.append("殖利率不足")
    _, fin_warning, fin_ok = financial_quality_score_and_warning(row, params)
    if params.use_fin_quality_filter and fin_warning:
        warnings.append(fin_warning)

    if not pd.isna(pe) and pe <= params.pe_deep_value_level and (revenue_bad or not no_new_low or (params.use_fin_quality_filter and not fin_ok)):
        return "D 類｜疑似價值陷阱", "、".join(warnings) or "低本益比但結構未穩"

    if not no_new_low and not pd.isna(dd60) and dd60 >= params.dd60_max:
        return "E 類｜結構偏弱", "、".join(warnings) or "跌深但尚未止跌"

    revenue_ok = (pd.isna(rev_yoy) or rev_yoy >= 0) and (pd.isna(acc_yoy) or acc_yoy >= 0)
    tech_stable = no_new_low and not pd.isna(support_distance) and support_distance <= params.support_distance_max

    if revenue_ok and tech_stable and not pd.isna(dd60) and dd60 >= params.dd60_min:
        return "C 類｜基本面錯殺型", "營收未明顯轉壞，技術進入左側觀察"

    if tech_stable and vol_shrink and not pd.isna(rsi) and rsi <= params.rsi_max:
        return "A 類｜低檔整理型", "量縮止跌，適合觀察二次回測"

    if not pd.isna(dd60) and dd60 >= 30 and not pd.isna(rsi) and rsi <= 35:
        return "B 類｜急跌反彈型", "跌深反彈機會，但部位要小"

    if warnings:
        return "觀察｜條件未完整", "、".join(warnings)
    return "觀察｜一般左側候選", "尚需等待更明確止跌或基本面確認"


def valuation_score(row: pd.Series, params: ScreenParams) -> int:
    score = 0
    pe = to_number(row.get("本益比"))
    pb = to_number(row.get("股價淨值比"))
    div_yield = to_number(row.get("殖利率%"))
    rev_yoy = sane_percent(row.get("月營收YoY%"))
    acc_yoy = sane_percent(row.get("累計營收YoY%"))
    rel_pe = to_number(row.get("相對產業本益比%"))

    if not pd.isna(pe) and 0 < pe <= params.pe_max:
        score += 10
        if pe <= min(params.pe_max * 0.75, 15):
            score += 3
    if not params.use_pb_filter or (not pd.isna(pb) and pb <= params.pb_max):
        score += 5
    if not params.use_dividend_filter or (not pd.isna(div_yield) and div_yield >= params.dividend_min):
        score += 5
    if not params.use_revenue_filter:
        score += 10
    else:
        if not pd.isna(rev_yoy) and rev_yoy >= params.revenue_yoy_min:
            score += 6
        elif pd.isna(rev_yoy) and not params.require_revenue_data:
            score += 3
        if not pd.isna(acc_yoy) and acc_yoy >= params.revenue_acc_yoy_min:
            score += 4
        elif pd.isna(acc_yoy) and not params.require_revenue_data:
            score += 2
    if not params.use_industry_pe_filter or (not pd.isna(rel_pe) and rel_pe <= params.industry_pe_ratio_max * 100):
        score += 5
    fin_score, _, _ = financial_quality_score_and_warning(row, params)
    score += fin_score
    return int(min(score, 45))


def analyze_symbol(df: pd.DataFrame, params: ScreenParams) -> Optional[Dict[str, object]]:
    if df.empty or len(df) < 80:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    current = close.iloc[-1]
    high20 = close.tail(20).max()
    high60 = close.tail(60).max()
    high120 = close.tail(120).max() if len(close) >= 120 else close.max()
    low20 = close.tail(20).min()
    low60 = close.tail(60).min()

    dd60 = (high60 - current) / high60 * 100 if high60 > 0 else np.nan
    dd120 = (high120 - current) / high120 * 100 if high120 > 0 else np.nan
    ret20 = (current / close.iloc[-21] - 1) * 100 if len(close) > 21 and close.iloc[-21] > 0 else np.nan
    ret60 = (current / close.iloc[-61] - 1) * 100 if len(close) > 61 and close.iloc[-61] > 0 else np.nan

    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    ma240 = close.rolling(240).mean()

    rsi = compute_rsi(close, 14)
    k, d = compute_kd(high, low, close, 9)

    rsi_now = float(rsi.iloc[-1])
    rsi_turn_up = bool(len(rsi) > params.rsi_turn_days and rsi.iloc[-1] > rsi.iloc[-1 - params.rsi_turn_days])

    kd_cross = False
    if len(k.dropna()) >= 2 and len(d.dropna()) >= 2:
        kd_cross = bool(k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] <= 35)

    avg_vol20_lots = float(volume.tail(20).mean() / 1000)

    support_name, support_price, support_distance = nearest_support(
        float(current),
        [
            ("60日低點", float(low60)),
            ("20MA", float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else np.nan),
            ("60MA", float(ma60.iloc[-1]) if not pd.isna(ma60.iloc[-1]) else np.nan),
            ("120MA", float(ma120.iloc[-1]) if not pd.isna(ma120.iloc[-1]) else np.nan),
            ("240MA", float(ma240.iloc[-1]) if not pd.isna(ma240.iloc[-1]) else np.nan),
        ],
    )

    no_new_low = True
    if len(close) >= 60:
        no_new_low = bool(close.tail(10).min() > close.tail(60).min() * 1.005)

    vol_shrink = bool(volume.tail(5).mean() <= volume.tail(20).mean() * 1.05)

    conditions = {
        "回檔區間": params.dd60_min <= dd60 <= params.dd60_max,
        "RSI低檔回升": rsi_now <= params.rsi_max and rsi_turn_up,
        "接近支撐": not np.isnan(support_distance) and support_distance <= params.support_distance_max,
        "均量足夠": avg_vol20_lots >= params.min_avg_volume_lots,
        "KD低檔金叉": kd_cross,
        "近期不破低": no_new_low,
    }

    tech_score = 0
    tech_score += 15 if conditions["回檔區間"] else 0
    tech_score += 15 if conditions["RSI低檔回升"] else 0
    tech_score += 15 if conditions["接近支撐"] else 0
    tech_score += 5 if conditions["均量足夠"] else 0
    tech_score += 5 if conditions["近期不破低"] else 0
    tech_score += 5 if conditions["KD低檔金叉"] else 0

    if params.require_kd_cross and not kd_cross:
        return None
    if params.require_no_new_low and not no_new_low:
        return None

    # 進場計畫：左側以支撐 / 二次回測 / 站回均線或小平台突破做三段。
    if pd.isna(support_price):
        support_price = float(low60)
        support_name = "60日低點"
    stop_loss = min(float(low20), float(low60), float(support_price)) * (1 - params.stop_loss_buffer / 100)
    test_entry_low = float(support_price) * 0.99
    test_entry_high = min(float(current), float(support_price) * 1.02) if current >= support_price else float(current)
    second_entry = max(float(support_price), float(low20))
    confirm_entry = max(
        float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else 0,
        float(high20),
    )
    first_resistance = first_notna(ma60.iloc[-1], ma120.iloc[-1], high60)
    second_resistance = float(high60)
    risk = max(float(current) - stop_loss, np.nan)
    reward = second_resistance - float(current)
    rr = reward / risk if risk and risk > 0 else np.nan

    if not pd.isna(rr) and rr < params.risk_reward_min:
        # 不直接淘汰，因為左側可能是低風險貼近支撐；但扣分。
        tech_score = max(0, tech_score - 5)

    return {
        "技術分數": int(tech_score),
        "收盤價_技術": round(float(current), 2),
        "60日回檔%": round(float(dd60), 2),
        "120日回檔%": round(float(dd120), 2),
        "RSI14": round(rsi_now, 2),
        "20日報酬%": round(float(ret20), 2) if not pd.isna(ret20) else np.nan,
        "60日報酬%": round(float(ret60), 2) if not pd.isna(ret60) else np.nan,
        "K值": round(float(k.iloc[-1]), 2) if not np.isnan(k.iloc[-1]) else np.nan,
        "D值": round(float(d.iloc[-1]), 2) if not np.isnan(d.iloc[-1]) else np.nan,
        "20日均量張": round(avg_vol20_lots, 0),
        "支撐名稱": support_name,
        "支撐參考價": round(float(support_price), 2),
        "支撐距離%": round(float(support_distance), 2) if not np.isnan(support_distance) else np.nan,
        "試單區間低": round(test_entry_low, 2),
        "試單區間高": round(test_entry_high, 2),
        "二次回測觀察價": round(float(second_entry), 2),
        "確認加碼價": round(float(confirm_entry), 2),
        "停損價": round(float(stop_loss), 2),
        "第一壓力價": round(float(first_resistance), 2) if not pd.isna(first_resistance) else np.nan,
        "第二壓力價": round(float(second_resistance), 2),
        "風報比估算": round(float(rr), 2) if not pd.isna(rr) and np.isfinite(rr) else np.nan,
        "MA20": round(float(ma20.iloc[-1]), 2) if not np.isnan(ma20.iloc[-1]) else np.nan,
        "MA60": round(float(ma60.iloc[-1]), 2) if not np.isnan(ma60.iloc[-1]) else np.nan,
        "MA120": round(float(ma120.iloc[-1]), 2) if not np.isnan(ma120.iloc[-1]) else np.nan,
        "MA240": round(float(ma240.iloc[-1]), 2) if not np.isnan(ma240.iloc[-1]) else np.nan,
        "KD低檔金叉": "是" if kd_cross else "否",
        "近期不破低": "是" if no_new_low else "否",
        "近5日量縮": "是" if vol_shrink else "否",
    }


def apply_base_filters(base: pd.DataFrame, params: ScreenParams) -> pd.DataFrame:
    """套用估值、P/B、殖利率、月營收、產業相對估值等濾網。"""
    df = base.copy()
    df = df[(df["本益比"] > 0) & (df["本益比"] <= params.pe_max)]

    if "收盤價" in df.columns and df["收盤價"].notna().any():
        df = df[(df["收盤價"].isna()) | (df["收盤價"] >= params.min_price)]

    if params.use_pb_filter:
        df = df[(df["股價淨值比"].isna()) | (df["股價淨值比"] <= params.pb_max)]

    if params.use_dividend_filter:
        df = df[(df["殖利率%"].isna()) | (df["殖利率%"] >= params.dividend_min)]

    if params.use_revenue_filter:
        if params.require_revenue_data:
            df = df[df["月營收YoY%"].notna() | df["累計營收YoY%"].notna()]
        yoy_ok = df["月營收YoY%"].isna() | (df["月營收YoY%"] >= params.revenue_yoy_min)
        acc_ok = df["累計營收YoY%"].isna() | (df["累計營收YoY%"] >= params.revenue_acc_yoy_min)
        if params.require_revenue_data:
            yoy_ok = df["月營收YoY%"].notna() & (df["月營收YoY%"] >= params.revenue_yoy_min)
            acc_ok = df["累計營收YoY%"].isna() | (df["累計營收YoY%"] >= params.revenue_acc_yoy_min)
        df = df[yoy_ok & acc_ok]

    if params.use_industry_pe_filter:
        rel = df["相對產業本益比%"]
        # 產業資料缺漏時不硬刪，避免 API 缺欄導致全空。
        df = df[rel.isna() | (rel <= params.industry_pe_ratio_max * 100)]

    return df.reset_index(drop=True)


def make_csv_download(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")


def mini_backtest(df: pd.DataFrame, params: ScreenParams, hold_days: int = 20, lookback_days: int = 180) -> pd.DataFrame:
    """單檔簡易技術回測：只測技術條件，不含歷史本益比與歷史月營收。"""
    if df.empty or len(df) < 120:
        return pd.DataFrame()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    rsi = compute_rsi(close, 14)
    k, d = compute_kd(high, low, close, 9)

    rows = []
    start = max(80, len(df) - lookback_days)
    end = len(df) - hold_days - 1
    last_signal_idx = -999
    for i in range(start, max(start, end)):
        # 避免訊號過度密集，至少隔 5 日。
        if i - last_signal_idx < 5:
            continue
        cur = float(close.iloc[i])
        high60 = float(close.iloc[max(0, i - 59): i + 1].max())
        low60 = float(close.iloc[max(0, i - 59): i + 1].min())
        dd60 = (high60 - cur) / high60 * 100 if high60 > 0 else np.nan
        ma120 = float(close.iloc[max(0, i - 119): i + 1].mean()) if i >= 119 else np.nan
        support_candidates = [("60日低點", low60)]
        if not pd.isna(ma120):
            support_candidates.append(("120MA", ma120))
        _, _, support_distance = nearest_support(cur, support_candidates)
        rsi_turn_up = bool(i > params.rsi_turn_days and rsi.iloc[i] > rsi.iloc[i - params.rsi_turn_days])
        kd_cross = bool(i >= 1 and k.iloc[i - 1] <= d.iloc[i - 1] and k.iloc[i] > d.iloc[i] and k.iloc[i] <= 35)
        no_new_low = bool(close.iloc[max(0, i - 9): i + 1].min() > close.iloc[max(0, i - 59): i + 1].min() * 1.005)
        avg_vol20_lots = float(volume.iloc[max(0, i - 19): i + 1].mean() / 1000)

        signal = (
            params.dd60_min <= dd60 <= params.dd60_max
            and rsi.iloc[i] <= params.rsi_max
            and rsi_turn_up
            and not pd.isna(support_distance)
            and support_distance <= params.support_distance_max
            and avg_vol20_lots >= params.min_avg_volume_lots
            and (kd_cross or not params.require_kd_cross)
            and (no_new_low or not params.require_no_new_low)
        )
        if not signal:
            continue
        future = close.iloc[i + 1: i + hold_days + 1]
        if future.empty:
            continue
        ret = (float(future.iloc[-1]) / cur - 1) * 100
        max_gain = (float(future.max()) / cur - 1) * 100
        max_dd = (float(future.min()) / cur - 1) * 100
        rows.append({
            "訊號日期": close.index[i].date().isoformat() if hasattr(close.index[i], "date") else str(close.index[i]),
            "訊號價": round(cur, 2),
            "持有天數": hold_days,
            "期末報酬%": round(ret, 2),
            "期間最大漲幅%": round(max_gain, 2),
            "期間最大回撤%": round(max_dd, 2),
            "當日RSI": round(float(rsi.iloc[i]), 2),
            "當日60日回檔%": round(float(dd60), 2),
        })
        last_signal_idx = i
    return pd.DataFrame(rows)


# ========= UI =========

if "watchlist" not in st.session_state:
    st.session_state.watchlist = pd.DataFrame()

st.title("📉 左側低估值選股器 Pro Max")
st.caption("本益比自訂 + 財報品質 + 大盤燈號 + 產業強弱 + 左側技術 + 部位計算 + 觀察名單狀態追蹤")

with st.expander("這個 App 的升級邏輯", expanded=False):
    st.markdown(
        """
        這版不是追強勢股，而是找：

        **估值合理偏低 → 月營收沒有明顯轉壞 → 股價跌深 → 接近支撐 → RSI 低檔回升 → 近期不再破底。**

        新增模組：
        - **P/B 濾網**：避免本益比低但資產評價過高。
        - **殖利率濾網**：適合成熟股、金融、傳產作為安全墊。
        - **月營收防呆**：避免低本益比但營收快速惡化。
        - **產業相對本益比**：避免絕對本益比低，但其實比同業貴。
        - **左側型態分類**：A 低檔整理、B 急跌反彈、C 基本面錯殺、D 價值陷阱、E 結構偏弱。
        - **自動交易計畫**：試單區、二次回測觀察價、確認加碼價、停損價、壓力價。
        - **觀察名單**：可下載 CSV，下次再上傳。
        - **單檔簡易回測**：只測技術條件，不代表完整歷史估值回測。
        """
    )

with st.sidebar:
    st.header("篩選參數")

    market_option = st.multiselect("市場", ["上市", "上櫃"], default=["上市", "上櫃"])
    include_twse = "上市" in market_option
    include_tpex = "上櫃" in market_option

    st.subheader("資料源設定")
    data_source_mode = st.selectbox(
        "資料源模式",
        ["FinMind優先（建議）", "只用FinMind", "官方API優先", "只用官方API"],
        index=0,
        help="TWSE 官方 API 在 Streamlit Cloud 有時會回傳 HTML；建議使用 FinMind 優先模式。",
    )
    try:
        default_finmind_token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        default_finmind_token = ""
    finmind_token = st.text_input(
        "FinMind API Token，可留空",
        value=default_finmind_token,
        type="password",
        help="留空也可嘗試免費額度；若掃描較多或遇到限制，建議到 FinMind 申請 token 後填入。也可在 Streamlit secrets 設定 FINMIND_TOKEN。",
    )

    with st.expander("上市資料備援上傳（TWSE/FinMind 都失敗時使用）", expanded=False):
        st.caption("如果 Streamlit Cloud 抓不到證交所上市資料，可以先從瀏覽器下載官方 JSON/CSV，再在這裡上傳。至少需要『上市估值資料』；行情與月營收可選。")
        uploaded_twse_pe = st.file_uploader("上市估值資料 BWIBBU_ALL，JSON/CSV", type=["json", "csv", "txt"], key="twse_pe_upload")
        uploaded_twse_quote = st.file_uploader("上市行情資料 STOCK_DAY_ALL，JSON/CSV，可留空", type=["json", "csv", "txt"], key="twse_quote_upload")
        uploaded_twse_revenue = st.file_uploader("上市月營收資料 t187ap15_L，JSON/CSV，可留空", type=["json", "csv", "txt"], key="twse_revenue_upload")
        st.markdown(
            "資料網址：  \n"
            "1. 估值：https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL  \n"
            "2. 行情：https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL  \n"
            "3. 月營收：https://openapi.twse.com.tw/v1/opendata/t187ap15_L"
        )

    st.subheader("估值濾網")
    pe_max = st.number_input("本益比上限", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
    min_price = st.number_input("最低股價", min_value=0.0, max_value=1000.0, value=10.0, step=1.0)
    use_pb_filter = st.checkbox("啟用 P/B 上限", value=True)
    pb_max = st.number_input("P/B 上限", min_value=0.1, max_value=20.0, value=2.5, step=0.1)
    use_dividend_filter = st.checkbox("啟用殖利率下限", value=False)
    dividend_min = st.number_input("殖利率下限 %", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
    use_industry_pe_filter = st.checkbox("啟用產業相對本益比", value=False)
    industry_pe_ratio_max = st.slider("個股本益比不得高於產業平均的倍數", 0.3, 2.0, 0.8, step=0.1)

    st.subheader("月營收防呆")
    use_revenue_filter = st.checkbox("啟用月營收防呆", value=True)
    require_revenue_data = st.checkbox("沒有月營收資料者直接排除", value=False)
    revenue_yoy_min = st.slider("最近月營收 YoY 下限 %", -80.0, 80.0, -20.0, step=1.0)
    revenue_acc_yoy_min = st.slider("累計營收 YoY 下限 %", -80.0, 80.0, -15.0, step=1.0)

    st.subheader("財報品質濾網")
    use_fin_quality_filter = st.checkbox("啟用財報品質濾網", value=False)
    use_yahoo_financial_quality = st.checkbox("用 Yahoo 補財報品質資料，較慢", value=False)
    require_fin_quality_data = st.checkbox("沒有財報品質資料者直接排除", value=False)
    eps_min = st.number_input("近四季 EPS 下限", min_value=-50.0, max_value=200.0, value=0.0, step=0.5)
    eps_growth_min = st.slider("獲利成長率下限 %", -100.0, 200.0, -30.0, step=5.0)
    gross_margin_min = st.slider("毛利率下限 %", -50.0, 100.0, 0.0, step=1.0)
    operating_margin_min = st.slider("營業利益率下限 %", -50.0, 100.0, 0.0, step=1.0)
    roe_min = st.slider("ROE 下限 %", -50.0, 100.0, 8.0, step=1.0)
    debt_ratio_max = st.slider("負債比 / 負債權益比上限 %", 0.0, 500.0, 200.0, step=10.0)
    require_positive_ocf = st.checkbox("要求營業現金流為正", value=False)
    uploaded_financial_csv = st.file_uploader("上傳財報品質 CSV，可留空", type=["csv"], key="financial_quality_csv")

    st.subheader("大盤與產業濾網")
    use_market_risk = st.checkbox("啟用大盤風險燈號", value=True)
    market_risk_mode = st.selectbox("大盤紅燈處理", ["只提示", "紅燈只保留A/C類", "紅燈停止篩選"], index=0)
    use_industry_strength_filter = st.checkbox("啟用產業強弱濾網", value=False)
    min_industry_ret20 = st.slider("產業20日平均報酬下限 %", -30.0, 30.0, -5.0, step=1.0)
    min_industry_relative20 = st.slider("產業相對大盤20日下限 %", -30.0, 30.0, -3.0, step=1.0)
    min_industry_breadth = st.slider("產業20日上漲比例下限 %", 0.0, 100.0, 35.0, step=5.0)

    st.subheader("左側技術條件")
    dd60_min, dd60_max = st.slider("距 60 日高點回檔幅度 %", 0.0, 80.0, (15.0, 40.0), step=1.0)
    rsi_max = st.slider("RSI14 上限", 10.0, 70.0, 45.0, step=1.0)
    rsi_turn_days = st.slider("RSI 回升比較天數", 1, 5, 3, step=1)
    support_distance_max = st.slider("距支撐最大距離 %", 0.0, 20.0, 8.0, step=0.5)
    min_avg_volume_lots = st.number_input("20日均量至少幾張", min_value=0.0, max_value=100000.0, value=1000.0, step=100.0)

    st.subheader("風險控管")
    stop_loss_buffer = st.slider("停損緩衝 %", 0.0, 10.0, 2.0, step=0.5)
    risk_reward_min = st.slider("最低風報比提醒", 0.0, 5.0, 1.5, step=0.1)
    pe_deep_value_level = st.slider("低本益比陷阱警示門檻", 3.0, 15.0, 10.0, step=0.5)
    account_capital = st.number_input("帳戶資金，用於部位計算", min_value=10000.0, max_value=100000000.0, value=1000000.0, step=50000.0)
    risk_per_trade_pct = st.slider("單筆最大風險占資金 %", 0.1, 5.0, 1.0, step=0.1)
    max_position_pct = st.slider("單檔最高部位占資金 %", 1.0, 100.0, 20.0, step=1.0)
    tranche1_pct = st.slider("第一筆試單比例 %", 0.0, 100.0, 30.0, step=5.0)
    tranche2_pct = st.slider("第二筆回測比例 %", 0.0, 100.0, 30.0, step=5.0)
    tranche3_pct = max(0.0, 100.0 - tranche1_pct - tranche2_pct)
    st.caption(f"第三筆確認加碼比例自動為 {tranche3_pct:.0f}%")

    st.subheader("嚴格條件")
    require_kd_cross = st.checkbox("必須 KD 低檔黃金交叉", value=False)
    require_no_new_low = st.checkbox("必須最近 10 日不再創 60 日新低", value=True)
    min_total_score = st.slider("最低左側總分", 0, 100, 60, step=5)

    st.subheader("效能設定")
    max_scan = st.slider("最多下載 K 線檔數", 20, 1500, 300, step=20)
    history_period = st.selectbox("K 線期間", ["9mo", "12mo", "18mo", "24mo"], index=1)
    sort_before_scan = st.selectbox("下載前排序", ["今日成交量張高到低", "總分前置估算高到低", "本益比低到高", "代號排序"], index=0)

    st.subheader("自選股模式")
    manual_codes = st.text_area("只掃描這些代號，可留空。例：2330, 2317, 2454", value="")

    uploaded_watchlist = st.file_uploader("上傳舊觀察名單 CSV，可留空", type=["csv"], key="watchlist_csv")
    run = st.button("開始篩選", type="primary", use_container_width=True)

params = ScreenParams(
    pe_max=pe_max,
    min_price=min_price,
    use_pb_filter=use_pb_filter,
    pb_max=pb_max,
    use_dividend_filter=use_dividend_filter,
    dividend_min=dividend_min,
    use_revenue_filter=use_revenue_filter,
    require_revenue_data=require_revenue_data,
    revenue_yoy_min=revenue_yoy_min,
    revenue_acc_yoy_min=revenue_acc_yoy_min,
    use_fin_quality_filter=use_fin_quality_filter,
    require_fin_quality_data=require_fin_quality_data,
    eps_min=eps_min,
    eps_growth_min=eps_growth_min,
    gross_margin_min=gross_margin_min,
    operating_margin_min=operating_margin_min,
    roe_min=roe_min,
    debt_ratio_max=debt_ratio_max,
    require_positive_ocf=require_positive_ocf,
    use_industry_pe_filter=use_industry_pe_filter,
    industry_pe_ratio_max=industry_pe_ratio_max,
    use_industry_strength_filter=use_industry_strength_filter,
    min_industry_ret20=min_industry_ret20,
    min_industry_relative20=min_industry_relative20,
    min_industry_breadth=min_industry_breadth,
    min_avg_volume_lots=min_avg_volume_lots,
    dd60_min=dd60_min,
    dd60_max=dd60_max,
    rsi_max=rsi_max,
    rsi_turn_days=rsi_turn_days,
    support_distance_max=support_distance_max,
    require_kd_cross=require_kd_cross,
    require_no_new_low=require_no_new_low,
    min_total_score=min_total_score,
    stop_loss_buffer=stop_loss_buffer,
    risk_reward_min=risk_reward_min,
    pe_deep_value_level=pe_deep_value_level,
    market_risk_mode=market_risk_mode,
)

if uploaded_watchlist is not None:
    try:
        st.session_state.watchlist = pd.read_csv(uploaded_watchlist)
        st.sidebar.success(f"已載入觀察名單：{len(st.session_state.watchlist):,} 筆")
    except Exception as exc:
        st.sidebar.warning(f"觀察名單讀取失敗：{exc}")

if not include_twse and not include_tpex:
    st.info("請至少選擇一個市場。")
    st.stop()

if run:
    start_time = time.time()

    market_info = {"燈號": "未啟用", "20日報酬%": np.nan, "說明": ""}
    if use_market_risk:
        with st.spinner("計算加權指數大盤風險燈號……"):
            market_info = load_market_risk(period=history_period)
        light = market_info.get("燈號", "未知")
        msg = f"大盤風險燈號：{light}｜{market_info.get('說明', '')}"
        if light == "綠燈":
            st.success(msg)
        elif light == "黃燈":
            st.warning(msg)
        elif light == "紅燈":
            st.error(msg)
            if params.market_risk_mode == "紅燈停止篩選":
                st.stop()
        else:
            st.info(msg)

    with st.spinner("下載估值、行情與月營收資料中……"):
        base = pd.DataFrame()
        finmind_first = data_source_mode in ["FinMind優先（建議）", "只用FinMind"]
        official_allowed = data_source_mode in ["FinMind優先（建議）", "官方API優先", "只用官方API"]
        finmind_allowed = data_source_mode in ["FinMind優先（建議）", "官方API優先", "只用FinMind"]

        if finmind_first and finmind_allowed:
            base = load_finmind_data(include_twse, include_tpex, include_revenue=use_revenue_filter, token=finmind_token)
            if not base.empty:
                st.success(f"已使用 FinMind 自動資料源取得資料：{len(base):,} 筆")

        if base.empty and official_allowed:
            base = load_official_data(include_twse, include_tpex, include_revenue=use_revenue_filter)
            if not base.empty:
                st.info(f"已使用官方 API 資料源取得資料：{len(base):,} 筆")

        if base.empty and (not finmind_first) and finmind_allowed:
            base = load_finmind_data(include_twse, include_tpex, include_revenue=use_revenue_filter, token=finmind_token)
            if not base.empty:
                st.success(f"官方 API 失敗後，已改用 FinMind 取得資料：{len(base):,} 筆")

    uploaded_twse_base = build_uploaded_twse_data(
        uploaded_twse_pe,
        uploaded_twse_quote,
        uploaded_twse_revenue,
        include_revenue=use_revenue_filter,
    )
    if not uploaded_twse_base.empty:
        if base.empty:
            base = uploaded_twse_base.copy()
        else:
            # 若官方上市資料也有部分成功，以上傳資料覆蓋同代號上市資料。
            uploaded_codes = set(uploaded_twse_base["代號"].astype(str))
            base = base[~((base["市場"] == "上市") & (base["代號"].astype(str).isin(uploaded_codes)))]
            base = pd.concat([base, uploaded_twse_base], ignore_index=True)
        st.success(f"已套用上傳的上市備援資料：{len(uploaded_twse_base):,} 筆")

    if base.empty:
        st.error("沒有取得資料。可能是 FinMind / 官方 API 暫時無法連線、API 額度受限，或回傳格式有變。可以先填入 FinMind token、改成「官方API優先」，或使用左側的備援上傳。")
        st.stop()

    # 手動股票代號模式先套用，避免全市場過濾後找不到自選股。
    manual_set = set()
    if manual_codes.strip():
        manual_set = {clean_code(x) for x in manual_codes.replace("\n", ",").split(",") if clean_code(x)}
        base = base[base["代號"].isin(manual_set)]

    fin_csv = normalize_financial_quality_upload(uploaded_financial_csv) if uploaded_financial_csv is not None else pd.DataFrame()
    if not fin_csv.empty:
        base = merge_financial_quality(base, fin_csv, pd.DataFrame())

    base["前置估值基本分"] = base.apply(lambda row: valuation_score(row, params), axis=1)
    filtered = apply_base_filters(base, params)

    if filtered.empty:
        st.warning("估值 / P/B / 殖利率 / 月營收濾網後沒有股票。可以放寬本益比、P/B、月營收 YoY 或取消強制月營收資料。")
        st.stop()

    # 財報品質資料若啟用 Yahoo 補充，先只抓通過估值與營收濾網後的前 max_scan 檔，避免手機版過慢。
    if params.use_fin_quality_filter and use_yahoo_financial_quality:
        preliminary = filtered.sort_values(["前置估值基本分", "本益比"], ascending=[False, True]).head(max_scan)
        with st.spinner("用 Yahoo 補財報品質資料中，這一步可能較慢……"):
            yahoo_fin = load_yahoo_financial_quality(tuple(preliminary["yfinance代號"].dropna().unique().tolist()))
        filtered = merge_financial_quality(filtered, fin_csv, yahoo_fin)

    filtered = apply_financial_quality_filters(filtered, params)

    if filtered.empty:
        st.warning("財報品質濾網後沒有股票。可以放寬 EPS、ROE、毛利率、營益率、負債比，或取消『沒有財報品質資料者直接排除』。")
        st.stop()

    filtered["前置估值基本分"] = filtered.apply(lambda row: valuation_score(row, params), axis=1)

    if sort_before_scan == "今日成交量張高到低":
        filtered = filtered.sort_values(["今日成交量張", "前置估值基本分", "本益比"], ascending=[False, False, True], na_position="last")
    elif sort_before_scan == "總分前置估算高到低":
        filtered = filtered.sort_values(["前置估值基本分", "本益比"], ascending=[False, True])
    elif sort_before_scan == "本益比低到高":
        filtered = filtered.sort_values("本益比", ascending=True)
    else:
        filtered = filtered.sort_values("代號")

    scan_df = filtered.head(max_scan).copy()
    tickers = scan_df["yfinance代號"].dropna().unique().tolist()

    st.write(f"估值與基本面濾網後共有 **{len(filtered):,}** 檔，這次下載前 **{len(tickers):,}** 檔 K 線做左側條件判斷。")

    with st.spinner("下載歷史 K 線並計算 RSI、KD、回檔、支撐與交易計畫……"):
        hist = download_history(tickers, period=history_period)

    rows = []
    progress = st.progress(0)
    for i, row in enumerate(scan_df.itertuples(index=False), start=1):
        ticker = getattr(row, "yfinance代號")
        one = extract_one_symbol(hist, ticker)
        result = analyze_symbol(one, params)
        if result is not None:
            base_info = row._asdict()
            base_info.update(result)
            base_info["估值基本分"] = valuation_score(pd.Series(base_info), params)
            base_info["左側總分"] = int(min(100, base_info["估值基本分"] + base_info["技術分數"]))
            setup_type, warning = classify_setup(base_info, params)
            base_info["左側型態"] = setup_type
            base_info["風險警示"] = warning
            if base_info["左側總分"] >= params.min_total_score:
                rows.append(base_info)
        progress.progress(i / max(len(scan_df), 1))
    progress.empty()

    result_df = pd.DataFrame(rows)
    elapsed = time.time() - start_time

    if result_df.empty:
        st.warning("沒有股票通過左側條件。可以降低最低總分、放寬 RSI 上限、支撐距離或回檔幅度。")
        st.stop()

    market_ret20 = to_number(market_info.get("20日報酬%"))
    result_df = add_industry_strength_columns(result_df, market_ret20)
    result_df = apply_industry_strength_filter(result_df, params)
    if result_df.empty:
        st.warning("產業強弱濾網後沒有股票。可以放寬產業 20 日報酬、相對大盤或上漲比例門檻。")
        st.stop()

    if use_market_risk and market_info.get("燈號") == "紅燈" and params.market_risk_mode == "紅燈只保留A/C類":
        result_df = result_df[result_df["左側型態"].str.contains("A 類|C 類", regex=True, na=False)].reset_index(drop=True)
        if result_df.empty:
            st.warning("大盤紅燈時只保留 A/C 類後沒有候選股。")
            st.stop()

    result_df["交易狀態"] = result_df.apply(determine_trade_status, axis=1)
    result_df = add_position_plan_columns(
        result_df,
        account_capital=account_capital,
        risk_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
        tranche1_pct=tranche1_pct,
        tranche2_pct=tranche2_pct,
        tranche3_pct=tranche3_pct,
    )

    result_df = result_df.sort_values(["左側總分", "左側型態", "本益比"], ascending=[False, True, True])

    show_cols = [
        "市場", "代號", "名稱", "左側總分", "估值基本分", "技術分數", "左側型態", "風險警示",
        "本益比", "股價淨值比", "殖利率%", "相對產業本益比%", "產業別",
        "月營收YoY%", "累計營收YoY%", "月營收MoM%", "營收年度", "營收月份",
        "近四季EPS", "EPS年增%", "毛利率%", "營業利益率%", "ROE%", "營業現金流", "負債比%", "財報品質分", "財報品質警示", "財報來源",
        "收盤價_技術", "60日回檔%", "120日回檔%", "20日報酬%", "60日報酬%", "個股相對大盤20日%", "RSI14", "K值", "D值",
        "產業20日平均報酬%", "產業60日平均報酬%", "產業相對大盤20日%", "產業20日上漲比例%",
        "20日均量張", "支撐名稱", "支撐參考價", "支撐距離%", "KD低檔金叉", "近期不破低", "近5日量縮",
        "交易狀態", "試單區間低", "試單區間高", "二次回測觀察價", "確認加碼價", "停損價", "第一壓力價", "第二壓力價", "風報比估算",
        "單股風險", "建議總股數", "建議總張數", "建議部位金額", "第一筆張數", "第二筆張數", "第三筆張數",
        "MA20", "MA60", "MA120", "MA240", "估值資料日期",
    ]
    show_cols = [c for c in show_cols if c in result_df.columns]

    st.success(f"完成，共找到 {len(result_df):,} 檔候選股。耗時約 {elapsed:.1f} 秒。")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("候選股", f"{len(result_df):,}")
    c2.metric("平均本益比", f"{result_df['本益比'].mean():.2f}")
    c3.metric("平均60日回檔", f"{result_df['60日回檔%'].mean():.1f}%")
    c4.metric("平均RSI", f"{result_df['RSI14'].mean():.1f}")
    c5.metric("A/C類占比", f"{result_df['左側型態'].str.contains('A 類|C 類', regex=True).mean() * 100:.0f}%")

    if use_market_risk:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("大盤燈號", str(market_info.get("燈號", "未知")))
        m2.metric("加權指數", str(market_info.get("指數收盤", "--")))
        m3.metric("大盤20日報酬", f"{to_number(market_info.get('20日報酬%')):.2f}%" if not pd.isna(to_number(market_info.get('20日報酬%'))) else "--")
        m4.metric("大盤RSI", f"{to_number(market_info.get('RSI14')):.1f}" if not pd.isna(to_number(market_info.get('RSI14'))) else "--")
        m5.metric("產業濾網", "啟用" if params.use_industry_strength_filter else "未啟用")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["候選股清單", "交易計畫", "產業/大盤", "觀察名單", "單檔圖表 / 簡易回測"])

    with tab1:
        st.dataframe(result_df[show_cols], use_container_width=True, hide_index=True)
        st.download_button(
            label="下載完整篩選結果 CSV",
            data=make_csv_download(result_df[show_cols]),
            file_name="left_side_value_stock_candidates_pro.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab2:
        plan_cols = [
            "市場", "代號", "名稱", "左側總分", "左側型態", "交易狀態", "收盤價_技術", "支撐名稱", "支撐參考價",
            "試單區間低", "試單區間高", "二次回測觀察價", "確認加碼價", "停損價",
            "第一壓力價", "第二壓力價", "風報比估算", "單股風險", "建議總股數", "建議總張數",
            "建議部位金額", "第一筆張數", "第二筆張數", "第三筆張數", "風險警示",
        ]
        plan_cols = [c for c in plan_cols if c in result_df.columns]
        st.markdown("**分批邏輯：第一筆只試單；第二筆看二次回測不破；第三筆等站回均線或突破小平台。**")
        st.dataframe(result_df[plan_cols], use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("**大盤燈號用來決定要不要降低左側部位；產業強弱用來避免買到完全退潮族群。**")
        if use_market_risk:
            market_table = pd.DataFrame([market_info])
            st.dataframe(market_table, use_container_width=True, hide_index=True)
        industry_cols = [
            "市場", "產業別", "產業20日平均報酬%", "產業60日平均報酬%", "產業相對大盤20日%", "產業20日上漲比例%", "代號"
        ]
        industry_cols = [c for c in industry_cols if c in result_df.columns]
        if industry_cols and "產業別" in result_df.columns:
            industry_summary = result_df.groupby(["市場", "產業別"], dropna=False).agg(
                候選檔數=("代號", "count"),
                產業20日平均報酬=("20日報酬%", "mean"),
                產業60日平均報酬=("60日報酬%", "mean"),
                產業相對大盤20日=("個股相對大盤20日%", "mean"),
                產業20日上漲比例=("20日報酬%", lambda x: (x > 0).mean() * 100),
                平均左側分數=("左側總分", "mean"),
            ).reset_index()
            for c in ["產業20日平均報酬", "產業60日平均報酬", "產業相對大盤20日", "產業20日上漲比例", "平均左側分數"]:
                industry_summary[c] = industry_summary[c].round(2)
            industry_summary = industry_summary.sort_values(["平均左側分數", "產業相對大盤20日"], ascending=[False, False])
            st.dataframe(industry_summary, use_container_width=True, hide_index=True)
        else:
            st.info("目前沒有足夠的產業資料可彙整。")

    with tab4:
        pick_options = (result_df["代號"] + " " + result_df["名稱"] + "｜" + result_df["市場"] + "｜" + result_df["左側型態"]).tolist()
        picks = st.multiselect("選擇要加入觀察名單的股票", pick_options)
        if st.button("加入觀察名單", use_container_width=True):
            selected_codes = [p.split()[0] for p in picks]
            add_df = result_df[result_df["代號"].isin(selected_codes)][show_cols].copy()
            if st.session_state.watchlist.empty:
                st.session_state.watchlist = add_df
            else:
                st.session_state.watchlist = pd.concat([st.session_state.watchlist, add_df], ignore_index=True)
                st.session_state.watchlist = st.session_state.watchlist.drop_duplicates(subset=["代號", "市場"], keep="last")
            st.success(f"已加入 {len(add_df):,} 檔。")

        if not st.session_state.watchlist.empty:
            if st.button("用本次篩選結果更新觀察名單狀態", use_container_width=True):
                st.session_state.watchlist = update_watchlist_status(st.session_state.watchlist, result_df)
                st.success("已更新觀察名單的交易狀態、價格、停損與加碼參考。")
            st.dataframe(st.session_state.watchlist, use_container_width=True, hide_index=True)
            st.download_button(
                label="下載觀察名單 CSV",
                data=make_csv_download(st.session_state.watchlist),
                file_name="left_side_watchlist.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("清空觀察名單", use_container_width=True):
                st.session_state.watchlist = pd.DataFrame()
                st.rerun()
        else:
            st.info("目前觀察名單是空的。")

    with tab5:
        choices = (result_df["代號"] + " " + result_df["名稱"] + "｜" + result_df["市場"] + "｜" + result_df["左側型態"]).tolist()
        selected = st.selectbox("選一檔查看收盤價、均線、交易計畫與簡易回測", choices)
        selected_code = selected.split()[0]
        selected_row = result_df[result_df["代號"] == selected_code].iloc[0]
        selected_ticker = selected_row["yfinance代號"]
        one = extract_one_symbol(hist, selected_ticker)

        detail_cols = [
            "代號", "名稱", "市場", "左側總分", "左側型態", "風險警示", "本益比", "股價淨值比", "殖利率%",
            "月營收YoY%", "累計營收YoY%", "近四季EPS", "毛利率%", "營業利益率%", "ROE%", "財報品質警示",
            "收盤價_技術", "交易狀態", "試單區間低", "試單區間高",
            "二次回測觀察價", "確認加碼價", "停損價", "第一壓力價", "第二壓力價", "風報比估算",
            "建議總張數", "第一筆張數", "第二筆張數", "第三筆張數",
        ]
        detail_cols = [c for c in detail_cols if c in result_df.columns]
        st.dataframe(pd.DataFrame([selected_row[detail_cols]]), use_container_width=True, hide_index=True)

        if not one.empty:
            chart_df = pd.DataFrame(index=one.index)
            chart_df["收盤價"] = one["Close"]
            chart_df["MA20"] = one["Close"].rolling(20).mean()
            chart_df["MA60"] = one["Close"].rolling(60).mean()
            chart_df["MA120"] = one["Close"].rolling(120).mean()
            chart_df["MA240"] = one["Close"].rolling(240).mean()
            st.line_chart(chart_df.tail(180), use_container_width=True)

            st.markdown("### 單檔簡易技術回測")
            st.caption("這裡只用目前設定的技術條件回測，不含歷史本益比、歷史 P/B、歷史月營收，因此只能當作粗略參考。")
            b1, b2 = st.columns(2)
            hold_days = b1.slider("持有天數", 5, 60, 20, step=5)
            lookback_days = b2.slider("回看天數", 90, 360, 180, step=30)
            bt = mini_backtest(one, params, hold_days=hold_days, lookback_days=lookback_days)
            if bt.empty:
                st.info("這段期間沒有出現符合條件的技術訊號，或歷史資料不足。")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("訊號次數", f"{len(bt):,}")
                m2.metric("勝率", f"{(bt['期末報酬%'] > 0).mean() * 100:.1f}%")
                m3.metric("平均報酬", f"{bt['期末報酬%'].mean():.2f}%")
                m4.metric("平均最大回撤", f"{bt['期間最大回撤%'].mean():.2f}%")
                st.dataframe(bt, use_container_width=True, hide_index=True)
        else:
            st.warning("這檔股票沒有成功取得 K 線資料。")

else:
    st.info("調整左側交易參數後，按左側的「開始篩選」。")
    st.markdown(
        """
        **建議起始參數：**
        - 本益比上限：20
        - P/B 上限：2.5
        - 月營收 YoY 下限：-20%
        - 累計營收 YoY 下限：-15%
        - 60 日高點回檔：15%～40%
        - RSI14 上限：45
        - 支撐距離：8% 以內
        - 20 日均量：1000 張以上
        - 最低左側總分：60～70
        """
    )

st.divider()
st.caption("免責聲明：本工具僅供研究與自動化篩選，不構成買賣建議。資料可能延遲或錯誤，請自行承擔投資風險。")
