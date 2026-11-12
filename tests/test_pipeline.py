import csv
import json
import unittest
from pathlib import Path


class PipelineEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.info = json.loads((cls.root / "data/processed/dataset_info.json").read_text(encoding="utf-8"))

    def test_counts(self):
        expected = {"train": (6549, 13098), "validation": (1404, 2808), "test": (1404, 2808)}
        for name, values in expected.items():
            self.assertEqual((self.info["splits"][name]["source_hours"], self.info["splits"][name]["examples"]), values)

    def test_disjoint_ids_and_row_counts(self):
        sets = []
        for name in ("train", "validation", "test"):
            path = self.root / "data/processed" / f"{name}.csv"
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), self.info["splits"][name]["examples"])
            sets.append({row["record_id"] for row in rows})
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_temporal_boundaries(self):
        splits = self.info["splits"]
        self.assertLess(splits["train"]["date_end"], splits["validation"]["date_start"])
        self.assertLess(splits["validation"]["date_end"], splits["test"]["date_start"])


if __name__ == "__main__":
    unittest.main()
