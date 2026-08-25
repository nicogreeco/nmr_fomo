"""Convert canonical molecular structures to Morgan fingerprints."""

from data.schema import CanonicalRecord
from data.validation import (
    IncompatibleRecordError,
    ValidationIssue,
    validate_canonical_record,
)

from . import ModelBatch, read_record


MORGAN_RADIUS = 2
MORGAN_BITS = 2048


class MorganProcessor:
    """Generate 2,048-bit ECFP4 fingerprints from canonical SMILES."""

    model_name = "morgan"

    def __init__(self, mode: str = "canonical", strict: bool = True):
        if mode not in {"canonical", "native"}:
            raise ValueError("mode must be 'canonical' or 'native'")
        self.mode = mode
        self.strict = strict

        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError as error:
            raise ImportError("Morgan processing requires RDKit") from error

        self._chem = Chem
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS,
            fpSize=MORGAN_BITS,
        )

    def _molecule(self, record: CanonicalRecord):
        smiles = record.smiles_canonical
        if not isinstance(smiles, str) or not smiles.strip():
            issue = ValidationIssue(
                "missing_smiles_canonical",
                "smiles_canonical",
                "Morgan fingerprints require a non-empty canonical SMILES",
            )
            raise IncompatibleRecordError(record.record_id, [issue])

        molecule = self._chem.MolFromSmiles(smiles)
        if molecule is None:
            issue = ValidationIssue(
                "invalid_smiles_canonical",
                "smiles_canonical",
                "RDKit could not parse the canonical SMILES",
            )
            raise IncompatibleRecordError(record.record_id, [issue])
        return molecule

    def prepare_record(self, value: object) -> dict:
        record = read_record(value)
        if self.strict:
            validation = validate_canonical_record(record)
            if validation.issues:
                raise IncompatibleRecordError(record.record_id, validation.issues)

        molecule = self._molecule(record)
        fingerprint = self._generator.GetFingerprint(molecule)
        return {
            "record_id": record.record_id,
            "fingerprint": tuple(int(bit) for bit in fingerprint),
        }

    def collate(self, records: list[dict]) -> ModelBatch:
        if not records:
            raise ValueError("cannot collate an empty Morgan batch")
        return ModelBatch(
            record_ids=[record["record_id"] for record in records],
            inputs={
                "fingerprints": [record["fingerprint"] for record in records]
            },
            metadata={
                "model_name": self.model_name,
                "modality": "molecule",
                "processor_mode": self.mode,
                "strict": self.strict,
                "radius": MORGAN_RADIUS,
                "n_bits": MORGAN_BITS,
            },
        )

    def __call__(self, records) -> ModelBatch:
        prepared_records = [self.prepare_record(record) for record in records]
        return self.collate(prepared_records)


__all__ = ["MORGAN_BITS", "MORGAN_RADIUS", "MorganProcessor"]
