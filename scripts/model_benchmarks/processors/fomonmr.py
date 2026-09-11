"""Adapt canonical records to the FoMoNMR foundation-model batch."""

from . import read_record


class FoMoNMRProcessor:
    """Use the foundation-model processor through the benchmark interface."""

    model_name = "fomonmr"

    def __init__(self, mode="canonical", strict=True):
        from model.processor import FoundationNMRProcessor

        self.mode = mode
        self.strict = strict
        self.processor = FoundationNMRProcessor()

    def prepare_record(self, value):
        record = read_record(value)
        return {"record_id": record.record_id, "record": record}

    def collate(self, records):
        return self.processor([record["record"] for record in records])

    def __call__(self, records):
        return self.collate([self.prepare_record(record) for record in records])


__all__ = ["FoMoNMRProcessor"]
