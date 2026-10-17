import ast
from pathlib import Path
import unittest

from watch import descendants

ROOT = Path(__file__).resolve().parent


class Contracts(unittest.TestCase):
    def test_real_descendants_not_PGID(self):
        processes = {10: {'ppid': 0, 'starttick': 1}, 20: {'ppid': 10, 'starttick': 2},
            30: {'ppid': 20, 'starttick': 3}, 40: {'ppid': 5, 'starttick': 4}}
        known = {}
        self.assertEqual(set(descendants(processes, 10, known)), {10, 20, 30})
        self.assertNotIn(40, known)

    def test_known_reparented_descendant_retained(self):
        processes = {20: {'ppid': 1, 'starttick': 2}, 30: {'ppid': 20, 'starttick': 3}}
        self.assertEqual(set(descendants(processes, None, {20: 2})), {20, 30})

    def test_reused_PID_not_owned(self):
        self.assertEqual(descendants({20: {'ppid': 1, 'starttick': 9}}, None, {20: 2}), {})

    def test_empty_before_suite(self):
        self.assertEqual(descendants({}, None, {}), {})

    def test_AST_and_no_GPU_monitor(self):
        for name in ('prepare.py', 'supervise.py', 'watch.py'):
            ast.parse((ROOT / name).read_text())
        text = (ROOT / 'watch.py').read_text()
        self.assertNotIn('import torch', text)
        self.assertNotIn('torch.xpu', text)

    def test_caps_and_no_auto_retry(self):
        text = (ROOT / 'prepare.py').read_text()
        for literal in ('96 * 1024**3', '128 * 1024**3', '12 * 3600', '380 * 1024**3', 'epoch + 120'):
            self.assertIn(literal, text)
        self.assertNotIn('while launch', (ROOT / 'prepare.py').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
