import unittest

from common.gpu_budget import parse_account_gpu_usage


class GpuBudgetTest(unittest.TestCase):
    def test_counts_distinct_gpus_owned_by_current_account(self) -> None:
        owners = {10: 1000, 11: 1000, 12: 2000}
        output = "10, GPU-A\n11, GPU-B\n12, GPU-C\n10, GPU-A\n"
        used = parse_account_gpu_usage(
            output,
            current_uid=1000,
            uid_for_pid=lambda pid: owners.get(pid),
        )
        self.assertEqual(used, {"GPU-A", "GPU-B"})


if __name__ == "__main__":
    unittest.main()
