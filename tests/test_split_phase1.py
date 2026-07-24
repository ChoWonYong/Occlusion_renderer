import unittest

from data.split_phase1 import make_split


class Phase1SplitTest(unittest.TestCase):
    def test_explicit_sequence_split_is_complete_and_disjoint(self) -> None:
        sequences = [f"{index:04d}" for index in range(21)]
        train = ["0005", "0010", "0014", "0015", "0016", "0000", "0001", "0002", "0003", "0004", "0006", "0007"]
        evaluation = ["0008", "0009", "0011", "0012", "0013", "0017", "0018", "0019", "0020"]
        first = make_split(sequences, train, evaluation)
        second = make_split(list(reversed(sequences)), train, evaluation)
        self.assertEqual(first, second)
        self.assertEqual(first["train_sequences"], train)
        self.assertEqual(first["eval_sequences"], evaluation)
        self.assertEqual(len(first["train_sequences"]), 12)
        self.assertEqual(len(first["eval_sequences"]), 9)
        self.assertFalse(set(first["train_sequences"]) & set(first["eval_sequences"]))
        self.assertEqual(first["crop_allowed_sequences"], first["train_sequences"])


if __name__ == "__main__":
    unittest.main()
