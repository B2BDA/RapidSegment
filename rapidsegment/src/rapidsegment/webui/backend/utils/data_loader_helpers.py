"""File loading helpers for Module 1 (mirrors 1_Data_Loader.py)."""
from __future__ import annotations

import os
import pyarrow.csv as pa_csv

from rapidsegment.utils.data_loader import UniversalDataLoader

from storage import DB_FILE, DB_FILE_MOD, load_state, save_state


NUMERIC = {"INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "REAL",
           "SMALLINT", "TINYINT", "HUGEINT", "INT", "UBIGINT", "UINTEGER"}


def is_num(t):
    return any(k in str(t).upper() for k in NUMERIC)


def detect_format(name):
    ext = os.path.splitext(name)[1].lower()
    return {
        ".csv": "CSV", ".tsv": "CSV (tab-separated)",
        ".parquet": "Parquet", ".pq": "Parquet",
        ".arrow": "Arrow", ".feather": "Arrow / Feather",
        ".xlsx": "Excel", ".xls": "Excel (legacy)",
    }.get(ext, "Unknown")


def detect_encoding(path):
    with open(path, "rb") as fh:
        raw = fh.read(100_000)
    for enc in ("utf-8", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


SAMPLE_NAMES = ["bank-full.csv", "train.csv"]


def find_sample_datasets():
    import rapidsegment
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(rapidsegment.__file__)))
    bases = [pkg_dir, os.path.join(os.getcwd(), "data"), os.path.expanduser("~/Downloads")]
    found = {}
    for base in bases:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f in SAMPLE_NAMES and f not in found:
                    p = os.path.join(root, f)
                    if os.path.getsize(p) > 0:
                        found[f] = p
    return found


def smart_default_hint():
    hints = []
    for base in (os.path.join(os.getcwd(), "data"), os.path.expanduser("~/Downloads")):
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            if detect_format(f) != "Unknown":
                hints.append(os.path.join(base, f))
    return hints[:3]


def load_file_udl(path, encoding="Auto-detect"):
    ext = os.path.splitext(path)[1].lower()
    if encoding != "Auto-detect" and ext == ".csv":
        enc = {"UTF-8": "utf8", "Latin-1": "latin1"}.get(encoding, "utf8")
        table = pa_csv.read_csv(path, read_options=pa_csv.ReadOptions(encoding=enc))
        return UniversalDataLoader().load(fallback_data=table)
    try:
        return UniversalDataLoader(file_path=path).load()
    except Exception:
        if ext == ".csv":
            last = None
            for enc in ("utf-8", "latin-1"):
                try:
                    table = pa_csv.read_csv(path, read_options=pa_csv.ReadOptions(encoding=enc))
                    return UniversalDataLoader().load(fallback_data=table)
                except Exception as exc:
                    last = exc
            if last is not None:
                raise last
        raise


def persist_file_direct(path, encoding=None, dataset_name=None):
    """Stream a file into the raw DuckDB table without a full in-RAM copy.

    A fresh raw load invalidates any previously materialized modified copy.
    """
    try:
        if os.path.exists(DB_FILE_MOD):
            os.remove(DB_FILE_MOD)
    except Exception:
        pass
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".csv", ".tsv", ".parquet", ".pq", ".arrow", ".feather"):
        UniversalDataLoader().stream_to_duckdb(DB_FILE, path, encoding)
    else:
        data = load_file_udl(path, encoding)
        import duckdb
        con = duckdb.connect(DB_FILE)
        con.execute("DROP TABLE IF EXISTS udl_data")
        con.execute("CREATE TABLE udl_data AS SELECT * FROM data")
        con.close()
    state = load_state()
    state["loaded"] = True
    state["tinfo"] = None
    state["data_modified"] = False
    if dataset_name:
        state["dataset_name"] = dataset_name
    save_state(state)
