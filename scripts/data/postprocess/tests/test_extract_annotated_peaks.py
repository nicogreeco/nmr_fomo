from __future__ import annotations

import unittest

try:
    from data.postprocess.extract_annotated_peaks import (
        find_exact_matches,
        full_inchikey,
    )
except ModuleNotFoundError:
    find_exact_matches = None
    full_inchikey = None


@unittest.skipUnless(full_inchikey is not None, "requires RDKit")
class ExtractAnnotatedPeaksTest(unittest.TestCase):
    def test_exact_full_inchikey_matching_expands_property_rows(self):
        ethanol_key = full_inchikey("CCO")
        assert ethanol_key is not None
        record_ids_by_key = {ethanol_key: ["nmr-1", "nmr-2"]}
        property_rows = [
            {"Drug": "CCO", "Y": "1.0"},
            {"Drug": "CCO", "Y": "2.0"},
            {"Drug": "CCC", "Y": "3.0"},
            {"Drug": "not-a-smiles", "Y": "4.0"},
        ]

        (
            expanded_rows,
            matched_record_ids,
            matched_molecule_keys,
            property_molecule_keys,
            invalid_or_unkeyed_rows,
            matched_property_rows,
        ) = find_exact_matches(property_rows, "Drug", record_ids_by_key)

        self.assertEqual(
            [row["record_id"] for row in expanded_rows],
            ["nmr-1", "nmr-2", "nmr-1", "nmr-2"],
        )
        self.assertEqual(matched_record_ids, {"nmr-1", "nmr-2"})
        self.assertEqual(matched_molecule_keys, {ethanol_key})
        self.assertEqual(property_molecule_keys, {ethanol_key, full_inchikey("CCC")})
        self.assertEqual(invalid_or_unkeyed_rows, 1)
        self.assertEqual(matched_property_rows, 2)


if __name__ == "__main__":
    unittest.main()
