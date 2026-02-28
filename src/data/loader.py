"""Data loading, column alignment, and metadata extraction.

Handles CIC-IDS-2017 and CSE-CIC-IDS-2018 CSV files with different
column naming conventions. Extracts session metadata (IPs, protocol,
timestamp) before dropping non-numeric columns.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def canonical_key(name: str) -> str:
    """Normalise a column name for fuzzy matching across datasets."""
    x = str(name).strip().lower()
    x = x.replace("/", " ")
    x = re.sub(r"[_\-]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    _abbrevs = {
        "dst": "destination", "src": "source",
        "byts": "bytes", "pkts": "packets", "pkt": "packet",
        "cnt": "count", "len": "length", "avg": "average",
        "tot": "total", "fwd": "forward", "bwd": "backward",
        "init": "initial", "win": "window", "seg": "segment",
        "hdr": "header", "iat": "inter arrival time",
    }
    tokens = x.split()
    tokens = [_abbrevs.get(t, t) for t in tokens]
    return " ".join(tokens)


def build_column_mapper(
    reference_cols: list[str], target_cols: list[str]
) -> dict[str, str]:
    """Build a mapping from *target* column names to *reference* column names.

    Uses exact-case-insensitive match first, then canonical_key fuzzy match.
    """
    key_to_ref: dict[str, str] = {}
    ambiguous: set[str] = set()
    for col in reference_cols:
        key = canonical_key(col)
        if key in key_to_ref and key_to_ref[key] != col:
            ambiguous.add(key)
        else:
            key_to_ref[key] = col
    for key in ambiguous:
        key_to_ref.pop(key, None)

    ref_lower = {c.strip().lower(): c for c in reference_cols}
    mapper: dict[str, str] = {}
    for col in target_cols:
        low = col.strip().lower()
        if low in ref_lower:
            mapper[col] = ref_lower[low]
            continue
        key = canonical_key(col)
        if key in key_to_ref:
            mapper[col] = key_to_ref[key]
    return mapper


def list_csv_files(directory: str | Path) -> list[Path]:
    """List all CSV files in a directory, sorted by name."""
    d = Path(directory)
    if not d.is_dir():
        logger.warning("Directory does not exist: %s", d)
        return []
    files = sorted(d.glob("*.csv"))
    logger.info("Found %d CSV files in %s", len(files), d)
    return files


def detect_label_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Find the label column from a list of candidates."""
    col_lower = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.strip().lower() in col_lower:
            return col_lower[cand.strip().lower()]
    return None


def _resolve_session_col(
    columns: list[str], aliases: list[str]
) -> Optional[str]:
    """Find a session-relevant column by checking aliases."""
    col_lower = {c.strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.strip().lower() in col_lower:
            return col_lower[alias.strip().lower()]
    return None


def extract_session_metadata(
    df: pd.DataFrame, cfg_session: dict
) -> pd.DataFrame:
    """Extract session columns and create session_id.

    Returns a DataFrame with columns: session_id, timestamp (numeric),
    plus the original index, for later windowing.
    """
    cols = list(df.columns)
    src_ip_col = _resolve_session_col(cols, cfg_session.get("src_ip", []))
    dst_ip_col = _resolve_session_col(cols, cfg_session.get("dst_ip", []))
    proto_col = _resolve_session_col(cols, cfg_session.get("protocol", []))
    ts_col = _resolve_session_col(cols, cfg_session.get("timestamp", []))

    meta = pd.DataFrame(index=df.index)

    if src_ip_col and dst_ip_col and proto_col:
        meta["session_id"] = (
            df[src_ip_col].astype(str) + "|"
            + df[dst_ip_col].astype(str) + "|"
            + df[proto_col].astype(str)
        ).apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:12])
        logger.info(
            "Session IDs built from: %s, %s, %s",
            src_ip_col, dst_ip_col, proto_col,
        )
    else:
        meta["session_id"] = "none"
        missing = []
        if not src_ip_col:
            missing.append("src_ip")
        if not dst_ip_col:
            missing.append("dst_ip")
        if not proto_col:
            missing.append("protocol")
        logger.warning(
            "Cannot build sessions — missing columns: %s. Using per-flow mode.",
            missing,
        )

    if ts_col:
        ts_raw = df[ts_col]
        ts_numeric = pd.to_numeric(ts_raw, errors="coerce")
        if ts_numeric.isna().all():
            try:
                ts_numeric = pd.to_datetime(ts_raw).astype(np.int64) // 10**9
            except Exception:
                ts_numeric = pd.Series(np.arange(len(df)), index=df.index)
                logger.warning("Timestamp parsing failed, using row index as proxy.")
        meta["timestamp"] = ts_numeric.values
    else:
        meta["timestamp"] = np.arange(len(df))
        logger.warning("No timestamp column found, using row index as proxy.")

    return meta


def read_csv_chunk(
    path: Path,
    chunksize: int = 50000,
    usecols: Optional[list[str]] = None,
    column_mapper: Optional[dict[str, str]] = None,
) -> pd.io.parsers.readers.TextFileReader:
    """Return a chunked CSV reader with optional column mapping applied."""
    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    return _MappedChunkReader(reader, column_mapper, usecols)


class _MappedChunkReader:
    """Wraps pandas chunk reader to apply column renaming per chunk."""

    def __init__(
        self,
        reader: pd.io.parsers.readers.TextFileReader,
        mapper: Optional[dict[str, str]],
        usecols: Optional[list[str]],
    ):
        self._reader = reader
        self._mapper = mapper or {}
        self._usecols = usecols

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:
        chunk = next(self._reader)
        chunk.columns = chunk.columns.str.strip()
        if self._mapper:
            chunk = chunk.rename(columns=self._mapper)
        if self._usecols:
            available = [c for c in self._usecols if c in chunk.columns]
            chunk = chunk[available]
        return chunk


def get_reference_columns(csv_files: list[Path]) -> list[str]:
    """Read column headers from the first file to use as reference."""
    if not csv_files:
        raise ValueError("No CSV files provided")
    cols = pd.read_csv(csv_files[0], nrows=0, encoding="utf-8").columns.str.strip().tolist()
    return cols


def compute_shared_features(
    cic_files: list[Path],
    cse_files: list[Path],
    drop_columns: list[str],
    label_candidates: list[str],
) -> tuple[list[str], str, str, dict[str, str]]:
    """Compute the intersection of numeric features between CIC and CSE.

    Returns: (shared_features, cic_label_col, cse_label_col, cse_mapper)
    """
    cic_cols = get_reference_columns(cic_files)
    cse_cols = get_reference_columns(cse_files)

    cic_label = detect_label_column(cic_cols, label_candidates)
    cse_mapper = build_column_mapper(cic_cols, cse_cols)
    mapped_cse_cols = [cse_mapper.get(c, c) for c in cse_cols]
    cse_label = detect_label_column(mapped_cse_cols, label_candidates)

    drop_lower = {d.strip().lower() for d in drop_columns}
    non_feature = set()
    if cic_label:
        non_feature.add(cic_label.strip().lower())
    for d in drop_lower:
        non_feature.add(d)

    def _numeric_features(cols: list[str]) -> set[str]:
        return {
            c for c in cols
            if c.strip().lower() not in non_feature
        }

    cic_feats = _numeric_features(cic_cols)
    cse_feats = _numeric_features(mapped_cse_cols)

    shared = sorted(cic_feats & cse_feats)
    if not shared:
        raise ValueError("No shared features between CIC-IDS2017 and CSE-CIC-IDS2018")

    logger.info(
        "Feature counts: CIC=%d, CSE=%d, shared=%d",
        len(cic_feats), len(cse_feats), len(shared),
    )
    return shared, cic_label or "Label", cse_label or "Label", cse_mapper
