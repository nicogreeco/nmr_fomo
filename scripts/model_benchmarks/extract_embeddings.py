"""Extract embeddings incrementally from canonical records."""

import argparse
import json
from itertools import islice
from pathlib import Path

from data import CanonicalNMRDataset, CanonicalParquetDataset, JsonlCanonicalReader
from data.validation import IncompatibleRecordError

from .factory import build_embedder, build_processor


MODEL_NAMES = (
    "nmrpeak",
    "nmrtrans",
    "ultranmr",
    "nmrsolver",
    "unimol2",
    "uni-mol2",
    "morgan",
)


def default_batch_size(model_name: str) -> int:
    return 1 if model_name in {"unimol2", "uni-mol2"} else 32


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "legacy_input",
        nargs="?",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--input",
        dest="input_option",
        type=Path,
        help="canonical Parquet or JSONL input",
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="records per model batch (default: 1 for UniMol2, otherwise 32)",
    )
    parser.add_argument(
        "--model-size", choices=("84M", "164M"), help="UniMol2 size"
    )
    parser.add_argument(
        "--mode",
        choices=("canonical", "native"),
        default="canonical",
        help="multiplicity mode (a no-op for shift-only and molecular models)",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--on-incompatible",
        choices=("error", "skip"),
        default="error",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        help="optional JSON report for skipped records",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="optional first-N limit for smoke tests",
    )
    parser.add_argument("--overwrite", action="store_true")

    arguments = parser.parse_args()
    if arguments.input_option and arguments.legacy_input:
        parser.error("give the input with --input or as a positional path, not both")
    arguments.input = arguments.input_option or arguments.legacy_input
    if arguments.input is None:
        parser.error("--input is required")
    if arguments.batch_size is None:
        arguments.batch_size = default_batch_size(arguments.model)
    if arguments.model in {"unimol2", "uni-mol2"}:
        if arguments.model_size is None:
            arguments.model_size = "84M"
    elif arguments.model_size is not None:
        parser.error("--model-size is only valid with UniMol2")
    return arguments


def check_arguments(arguments: argparse.Namespace) -> None:
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if arguments.max_records is not None and arguments.max_records < 1:
        raise ValueError("--max-records must be at least 1")
    if arguments.output.suffix.lower() != ".parquet":
        raise ValueError("--output must be a .parquet file")
    if not arguments.input.is_file():
        raise FileNotFoundError(f"canonical input not found: {arguments.input}")

    partial_output = Path(str(arguments.output) + ".partial")
    output_paths = [arguments.output, partial_output]
    if arguments.rejections:
        output_paths.append(arguments.rejections)
    input_path = arguments.input.resolve()
    if any(path.resolve() == input_path for path in output_paths):
        raise ValueError("output and rejection files must not overwrite the input")
    if (
        arguments.rejections
        and arguments.output.resolve() == arguments.rejections.resolve()
    ):
        raise ValueError("--output and --rejections must be different files")

    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not arguments.overwrite:
        raise FileExistsError(
            "refusing to overwrite existing file(s): "
            + ", ".join(existing)
            + "; pass --overwrite to replace them"
        )


def build_dataset(input_path: Path, max_records: int | None = None):
    """Open canonical data without applying model-specific transformations."""

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        dataset = CanonicalParquetDataset(input_path)
    elif suffix == ".jsonl":
        dataset = CanonicalNMRDataset(JsonlCanonicalReader(input_path))
    else:
        raise ValueError("input must be a canonical .parquet or .jsonl file")

    if max_records is None:
        return dataset
    return CanonicalNMRDataset(islice(dataset, max_records))


def _write_embedding_batch(writer, schema, pyarrow, embeddings, record_ids) -> None:
    """Write one model batch as one Parquet row group."""

    dimension = embeddings.shape[1]
    flat_values = pyarrow.array(
        embeddings.contiguous().numpy().reshape(-1),
        type=pyarrow.float32(),
    )
    embedding_column = pyarrow.FixedSizeListArray.from_arrays(
        flat_values, dimension
    )
    table = pyarrow.Table.from_arrays(
        [pyarrow.array(record_ids), embedding_column], schema=schema
    )
    writer.write_table(table)


def _issue_as_dict(issue) -> dict[str, str]:
    return {"code": issue.code, "path": issue.path, "message": issue.message}


def main() -> None:
    arguments = parse_arguments()
    check_arguments(arguments)

    import torch
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from torch.utils.data import DataLoader

    dataset = build_dataset(arguments.input, arguments.max_records)
    processor = build_processor(arguments.model, mode=arguments.mode, strict=True)

    embedder_options = {"device": arguments.device}
    if arguments.checkpoint is not None:
        embedder_options["checkpoint_path"] = arguments.checkpoint
    if arguments.model in {"unimol2", "uni-mol2"}:
        embedder_options["model_size"] = arguments.model_size
    embedder = build_embedder(arguments.model, **embedder_options)

    rejections = []

    def collate_records(records):
        if arguments.on_incompatible == "error":
            return processor(records)

        prepared_records = []
        for record in records:
            try:
                prepared_records.append(processor.prepare_record(record))
            except IncompatibleRecordError as error:
                rejections.append(
                    {
                        "record_id": error.record_id,
                        "issues": [_issue_as_dict(issue) for issue in error.issues],
                    }
                )

        # NMRPeak may discover an overlong token sequence during collation.
        while prepared_records:
            try:
                return processor.collate(prepared_records)
            except IncompatibleRecordError as error:
                rejections.append(
                    {
                        "record_id": error.record_id,
                        "issues": [_issue_as_dict(issue) for issue in error.issues],
                    }
                )
                prepared_records = [
                    record
                    for record in prepared_records
                    if record["record_id"] != error.record_id
                ]
        return None

    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_records,
    )

    dimension = int(embedder.dimension)
    schema = pa.schema(
        [
            pa.field("record_id", pa.string(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), dimension), nullable=False),
        ]
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = Path(str(arguments.output) + ".partial")
    writer = parquet.ParquetWriter(partial_output, schema, compression="zstd")

    accepted_count = 0
    last_result = None
    try:
        for batch in loader:
            if batch is None:
                continue
            result = embedder.encode(batch)
            embeddings = result.embeddings.detach().to(
                device="cpu", dtype=torch.float32
            )
            expected_shape = (len(result.record_ids), dimension)
            if tuple(embeddings.shape) != expected_shape:
                raise RuntimeError(
                    "embedder returned shape "
                    f"{tuple(embeddings.shape)}; expected {expected_shape}"
                )
            _write_embedding_batch(
                writer, schema, pa, embeddings, result.record_ids
            )
            accepted_count += len(result.record_ids)
            last_result = result

        metadata = {
            "model_name": embedder.model_name,
            "checkpoint": str(getattr(embedder, "checkpoint_path", "")) or None,
            "dimension": dimension,
            "pooling": embedder.pooling,
            "modality": getattr(embedder, "modality", "1H+13C"),
            "processor_mode": arguments.mode,
            "input_record_limit": arguments.max_records,
            "accepted_count": accepted_count,
            "rejected_count": len(rejections),
        }
        if hasattr(embedder, "model_size"):
            metadata["model_size"] = embedder.model_size

        if last_result is not None:
            metadata.update(last_result.metadata)
            metadata["checkpoint"] = last_result.checkpoint

        writer.add_key_value_metadata(
            {"nmr_embedding_metadata": json.dumps(metadata, sort_keys=True)}
        )
    finally:
        writer.close()

    # The final path appears only after extraction and footer writing succeed.
    partial_output.replace(arguments.output)

    if arguments.rejections:
        arguments.rejections.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_name": metadata["model_name"],
            "rejected_count": len(rejections),
            "rejections": rejections,
        }
        arguments.rejections.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    print(f"saved {accepted_count} embeddings to {arguments.output}")
    if rejections:
        print(f"skipped {len(rejections)} incompatible records")


if __name__ == "__main__":
    main()

