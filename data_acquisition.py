from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _dedupe_index(cptac_module) -> None:
    index = getattr(cptac_module, "INDEX", None)
    if index is None or not hasattr(index, "columns"):
        return
    if "filename" not in index.columns:
        return
    duplicates = index["filename"].duplicated(keep="first")
    if bool(duplicates.any()):
        cptac_module.INDEX = index.loc[~duplicates].copy()
        logging.info("Deduplicated cptac.INDEX by filename.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CPTAC datasets and save RNA/Protein tables to data/raw.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated list of dataset names to download (e.g., brca,pdac).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step and only try to load and export tables.",
    )
    return parser.parse_args()


def _list_datasets(cptac_module) -> Optional[Iterable[str]]:
    if hasattr(cptac_module, "list_datasets"):
        return cptac_module.list_datasets()
    if hasattr(cptac_module, "datasets") and hasattr(cptac_module.datasets, "list_datasets"):
        return cptac_module.datasets.list_datasets()
    return None


def _camel_case(name: str) -> str:
    parts = []
    current = []
    for ch in name:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                parts.append("".join(current))
                current = []
    if current:
        parts.append("".join(current))
    return "".join(part[:1].upper() + part[1:].lower() for part in parts)


def _download_dataset(cptac_module, name: str) -> None:
    if hasattr(cptac_module, "download"):
        try:
            cptac_module.download(dataset=name)
        except TypeError:
            cptac_module.download(name)
        return
    if hasattr(cptac_module, "download_dataset"):
        cptac_module.download_dataset(name)
        return
    raise RuntimeError("No download function found in cptac module.")


def _download_with_retry(cptac_module, name: str, max_attempts: int = 5) -> None:
    delay = 2
    for attempt in range(1, max_attempts + 1):
        try:
            _download_dataset(cptac_module, name)
            return
        except Exception as exc:
            message = str(exc)
            is_rate_limited = "429" in message or "Too Many Requests" in message
            if not is_rate_limited or attempt == max_attempts:
                raise
            logging.warning(
                "Rate limited downloading %s (attempt %s/%s). Retrying in %ss.",
                name,
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
            delay *= 2


def _wrap_cptac_download(cptac_module, max_attempts: int = 5) -> None:
    if not hasattr(cptac_module, "download"):
        return
    original = cptac_module.download

    def wrapped(*args, **kwargs):
        delay = 2
        for attempt in range(1, max_attempts + 1):
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                message = str(exc)
                is_rate_limited = "429" in message or "Too Many Requests" in message
                if not is_rate_limited or attempt == max_attempts:
                    raise
                logging.warning(
                    "Rate limited by Zenodo (attempt %s/%s). Retrying in %ss.",
                    attempt,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

    cptac_module.download = wrapped


def _load_dataset(cptac_module, name: str):
    if hasattr(cptac_module, "get_dataset"):
        return cptac_module.get_dataset(name)

    class_name = _camel_case(name)
    dataset_cls = getattr(cptac_module, class_name, None)
    if dataset_cls is None:
        raise RuntimeError(f"No dataset class found for '{name}' (expected {class_name}).")
    return dataset_cls()


def _get_sources(dataset_obj, data_type: str, cptac_module=None, dataset_name: Optional[str] = None) -> Optional[list[str]]:
    if not hasattr(dataset_obj, "list_data_sources"):
        return None
    try:
        sources_df = dataset_obj.list_data_sources()
    except Exception:
        return None
    if "Data type" not in sources_df.columns or "Available sources" not in sources_df.columns:
        return None
    matches = sources_df[sources_df["Data type"] == data_type]
    if matches.empty:
        return None
    sources = matches["Available sources"].iloc[0]
    if not sources:
        return None
    sources = list(sources)
    if cptac_module is None:
        return sources

    index = getattr(cptac_module, "INDEX", None)
    if index is None or not hasattr(index, "columns") or "filename" not in index.columns:
        return sources

    cancer_type = dataset_name or getattr(dataset_obj, "_cancer_type", None) or ""
    filenames = index["filename"].astype(str)

    valid_sources = []
    for source in sources:
        if source in ["harmonized", "mssm"]:
            prefix = f"{source}-all_cancers-{data_type}-"
        elif source == "washu" and data_type in ["tumor_purity", "hla_typing"]:
            prefix = f"{source}-all_cancers-{data_type}-"
        else:
            prefix = f"{source}-{cancer_type}-{data_type}-"
        if filenames.str.startswith(prefix).any():
            valid_sources.append(source)

    return valid_sources or sources


def _try_get_table(dataset_obj, data_type: str, method_names: Iterable[str], cptac_module=None, dataset_name: Optional[str] = None):
    sources = _get_sources(dataset_obj, data_type, cptac_module=cptac_module, dataset_name=dataset_name) or [None]
    for method_name in method_names:
        if not hasattr(dataset_obj, method_name):
            continue
        attr = getattr(dataset_obj, method_name)
        if callable(attr):
            for source in sources:
                try:
                    if source is not None:
                        result = attr(source=source)
                    else:
                        result = attr()
                except TypeError:
                    try:
                        result = attr()
                    except Exception:
                        continue
                except Exception:
                    continue
                if hasattr(result, "to_parquet"):
                    return result
            continue
        if hasattr(attr, "to_parquet"):
            return attr
    if hasattr(dataset_obj, "get_dataframe"):
        for source in sources:
            if source is None:
                continue
            try:
                result = dataset_obj.get_dataframe(data_type, source=source)
                if hasattr(result, "to_parquet"):
                    return result
            except Exception:
                continue
    return None


def main() -> int:
    _setup_logging()
    args = _parse_args()

    try:
        import cptac
    except ImportError:
        logging.error("cptac is not installed. Run: pip install -r requirements.txt")
        return 1
    _dedupe_index(cptac)
    _wrap_cptac_download(cptac)

    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        datasets = [name.strip() for name in args.only.split(",") if name.strip()]
    else:
        datasets = _list_datasets(cptac)
        if datasets is None:
            logging.error(
                "Could not detect dataset list. Use --only to specify datasets explicitly."
            )
            return 1
        if hasattr(datasets, "empty"):
            if datasets.empty:
                logging.error(
                    "No datasets returned. Use --only to specify datasets explicitly."
                )
                return 1
            if "Cancer" in datasets.columns:
                datasets = sorted(
                    {str(name).lower() for name in datasets["Cancer"].tolist() if name}
                )
            else:
                datasets = datasets.iloc[:, 0].tolist()

    summary_path = raw_dir / "download_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}

    for name in datasets:
        logging.info("Processing dataset: %s", name)
        if not args.skip_download:
            try:
                _download_with_retry(cptac, name)
            except Exception as exc:
                logging.warning(
                    "Download step failed for %s (will try to load anyway): %s",
                    name,
                    exc,
                )

        try:
            ds = _load_dataset(cptac, name)
        except Exception as exc:
            logging.warning("Load failed for %s: %s", name, exc)
            continue

        transcriptomics = _try_get_table(
            ds,
            "transcriptomics",
            ["get_transcriptomics", "get_rna", "get_rnaseq"],
            cptac_module=cptac,
            dataset_name=name,
        )
        proteomics = _try_get_table(
            ds,
            "proteomics",
            ["get_proteomics", "get_proteomics_TMT", "get_proteomics_tmt"],
            cptac_module=cptac,
            dataset_name=name,
        )

        dataset_dir = raw_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dataset_summary = {}

        if transcriptomics is not None:
            transcript_path = dataset_dir / "transcriptomics.parquet"
            transcriptomics.to_parquet(transcript_path)
            dataset_summary["transcriptomics"] = {
                "rows": int(transcriptomics.shape[0]),
                "cols": int(transcriptomics.shape[1]),
                "path": str(transcript_path),
            }
        else:
            logging.warning("No transcriptomics table for %s", name)

        if proteomics is not None:
            proteomics_path = dataset_dir / "proteomics.parquet"
            proteomics.to_parquet(proteomics_path)
            dataset_summary["proteomics"] = {
                "rows": int(proteomics.shape[0]),
                "cols": int(proteomics.shape[1]),
                "path": str(proteomics_path),
            }
        else:
            logging.warning("No proteomics table for %s", name)

        if dataset_summary:
            summary[name] = dataset_summary

    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote summary to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
