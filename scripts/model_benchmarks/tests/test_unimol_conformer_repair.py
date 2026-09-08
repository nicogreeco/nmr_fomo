"""Run in the UniMol2 environment; no checkpoint or real dataset required."""

import unittest
import importlib.util
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("unimol_tools"), "Requires UniMol2 environment")
class ConformerRepairTests(unittest.TestCase):
    def test_boron_record_is_repaired_without_changing_canonical_graph(self):
        from rdkit import Chem
        from unimol_tools.data.conformer import get_graph, get_graph_features
        from model.prepare_unimol_sidecars import restore_canonical_graph, singleton_batch
        from model_benchmarks.processors.unimol2 import UniMol2Processor
        from data.validation import IncompatibleRecordError

        row = {"record_id": "boron-regression", "smiles_canonical":
               "Cc1nc2c(c(-c3ccccc3N(C)C)c1C)[B-]1(Oc3ccc4ccccc4c3-c3c(ccc4ccccc34)O1)[n+]1c-2ccc2ccccc21"}
        with self.assertRaises(IncompatibleRecordError):
            UniMol2Processor(strict=False)([row])
        prepared = restore_canonical_graph(row)
        molecule = Chem.RemoveAllHs(Chem.AddHs(Chem.MolFromSmiles(row["smiles_canonical"])))
        nodes, edges, attributes = get_graph(molecule)
        expected = get_graph_features(attributes, edges, nodes, drop_feat=0,
                                      mask=np.ones(molecule.GetNumAtoms(), dtype=bool))
        for name in ("atom_feat", "edge_feat", "shortest_path", "degree"):
            np.testing.assert_array_equal(prepared["features"][name], expected[name])
        batch = singleton_batch([row])
        self.assertEqual(batch.record_ids, [row["record_id"]])
        self.assertTrue(np.isfinite(batch.inputs["src_coord"].numpy()).all())

    def test_normal_molecule_uses_identical_original_features(self):
        from model.prepare_unimol_sidecars import singleton_batch
        from model_benchmarks.processors.unimol2 import UniMol2Processor
        row = {"record_id": "ethanol", "smiles_canonical": "CCO"}
        original = UniMol2Processor(strict=False)([row])
        actual = singleton_batch([row])
        for name in original.inputs:
            np.testing.assert_array_equal(original.inputs[name], actual.inputs[name])


if __name__ == "__main__":
    unittest.main()
