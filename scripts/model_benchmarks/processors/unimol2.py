"""Convert canonical molecular structures to UniMol2 tensor batches."""

from data.schema import CanonicalRecord
from data.validation import (
    IncompatibleRecordError,
    ValidationIssue,
    validate_canonical_record,
)

from . import ModelBatch, read_record


FEATURE_NAMES = (
    "atom_feat",
    "atom_mask",
    "edge_feat",
    "shortest_path",
    "degree",
    "pair_type",
    "attn_bias",
    "src_tokens",
    "src_coord",
)


class UniMol2Processor:
    """Validate structures, generate conformers, and collate UniMol2 inputs.

    ``mode`` is accepted for the shared processor API but does not affect this
    structure-only model.
    """

    model_name = "unimol2"
    max_atoms = 128

    def __init__(self, mode: str = "canonical", strict: bool = True):
        if mode not in {"canonical", "native"}:
            raise ValueError("mode must be 'canonical' or 'native'")
        self.mode = mode
        self.strict = strict

        try:
            from rdkit import Chem
            from unimol_tools.data.conformer import UniMolV2Feature
        except ImportError as error:
            raise ImportError(
                "UniMol2 processing requires RDKit and unimol_tools from "
                "unimol2_venv"
            ) from error

        self._chem = Chem
        self._feature_maker = UniMolV2Feature(
            seed=42,
            max_atoms=self.max_atoms,
            mode="fast",
            remove_hs=True,
            multi_process=False,
        )

    def _model_issues(self, record: CanonicalRecord) -> list[ValidationIssue]:
        smiles = record.smiles_canonical
        if not isinstance(smiles, str) or not smiles.strip():
            return [
                ValidationIssue(
                    "missing_smiles_canonical",
                    "smiles_canonical",
                    "UniMol2 requires a non-empty canonical SMILES",
                )
            ]

        molecule = self._chem.MolFromSmiles(smiles)
        if molecule is None:
            return [
                ValidationIssue(
                    "invalid_smiles_canonical",
                    "smiles_canonical",
                    "RDKit could not parse the canonical SMILES",
                )
            ]
        if molecule.GetNumHeavyAtoms() > self.max_atoms:
            return [
                ValidationIssue(
                    "too_many_atoms",
                    "smiles_canonical",
                    f"UniMol2 supports at most {self.max_atoms} heavy atoms",
                )
            ]
        return []

    def prepare_record(self, value: object) -> dict:
        record = read_record(value)
        issues = []
        if self.strict:
            issues.extend(validate_canonical_record(record).issues)
        issues.extend(self._model_issues(record))
        if issues:
            raise IncompatibleRecordError(record.record_id, issues)

        try:
            features, _ = self._feature_maker.single_process(
                record.smiles_canonical
            )
        except Exception as error:
            issue = ValidationIssue(
                "conformer_generation_failed",
                "smiles_canonical",
                f"UniMol2 preprocessing failed: {error}",
            )
            raise IncompatibleRecordError(record.record_id, [issue]) from error

        missing = [name for name in FEATURE_NAMES if name not in features]
        if missing:
            raise RuntimeError(
                "UniMol2 preprocessing omitted features: " + ", ".join(missing)
            )
        return {
            "record_id": record.record_id,
            "features": {name: features[name] for name in FEATURE_NAMES},
        }

    def collate(self, records: list[dict]) -> ModelBatch:
        if not records:
            raise ValueError("cannot collate an empty UniMol2 batch")

        import torch
        from unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords

        feature_rows = [record["features"] for record in records]
        inputs = {
            "atom_feat": pad_coords(
                [torch.tensor(row["atom_feat"]) for row in feature_rows],
                pad_idx=0,
                dim=8,
            ),
            "atom_mask": pad_1d_tokens(
                [torch.tensor(row["atom_mask"]) for row in feature_rows],
                pad_idx=0,
            ),
            "edge_feat": pad_2d(
                [torch.tensor(row["edge_feat"]) for row in feature_rows],
                pad_idx=0,
                dim=3,
            ),
            "shortest_path": pad_2d(
                [torch.tensor(row["shortest_path"]) for row in feature_rows],
                pad_idx=0,
            ),
            "degree": pad_1d_tokens(
                [torch.tensor(row["degree"]) for row in feature_rows],
                pad_idx=0,
            ),
            "pair_type": pad_2d(
                [torch.tensor(row["pair_type"]) for row in feature_rows],
                pad_idx=0,
                dim=2,
            ),
            "attn_bias": pad_2d(
                [torch.tensor(row["attn_bias"]) for row in feature_rows],
                pad_idx=0,
            ),
            "src_tokens": pad_1d_tokens(
                [torch.tensor(row["src_tokens"]) for row in feature_rows],
                pad_idx=0,
            ),
            "src_coord": pad_coords(
                [torch.tensor(row["src_coord"]) for row in feature_rows],
                pad_idx=0,
            ),
        }

        return ModelBatch(
            record_ids=[record["record_id"] for record in records],
            inputs=inputs,
            metadata={
                "model_name": self.model_name,
                "modality": "molecule",
                "processor_mode": self.mode,
                "strict": self.strict,
                "conformer_seed": 42,
                "max_atoms": self.max_atoms,
            },
        )

    def __call__(self, records) -> ModelBatch:
        prepared_records = [self.prepare_record(record) for record in records]
        return self.collate(prepared_records)


__all__ = ["FEATURE_NAMES", "UniMol2Processor"]
