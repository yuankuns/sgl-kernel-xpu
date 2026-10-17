import ast
from decimal import Decimal
from pathlib import Path
import unittest

from audit_dependencies import fields, ns
from proposed_binding_probe import tensor_metadata

ROOT = Path(__file__).resolve().parent


class CPUContracts(unittest.TestCase):
    def test_actual_command_field_name_parse(self):
        body = '<TR><TD><B>zeCall</B>\n hCommandList</TD><TD>0x123</TD></TR>'
        self.assertEqual(fields(body)['hCommandList'], '0x123')

    def test_exact_time_domain_conversion(self):
        self.assertEqual(ns(Decimal('1340876703048.597'), 1790857026000000000), 1792197902703048597)

    def test_proposal_no_native_edit_fake_or_timer(self):
        text = (ROOT / 'proposed_binding_probe.py').read_text()
        ast.parse(text)
        for token in ('load_library', 'register_fake', 'ctypes', 'elapsed_time', 'Event(', '/proc/'):
            self.assertNotIn(token, text)

    def test_actual_audit_does_not_filter_small_negative_gaps(self):
        import json
        value = json.loads((ROOT / 'DEPENDENCY_AUDIT.json').read_text())
        for case in value.values():
            self.assertEqual(len(case['negative_observed_dependency_gaps']), 2)
            self.assertEqual(case['maximum_Chrome_raw_conversion_error_ns'], 0)
            self.assertFalse(case['last_reset_start_after_last_QK_end'])
            self.assertFalse(case['last_reset_end_after_last_QK_end'])
            self.assertTrue(case['dumped_edges_NOT_claimed_full_backend_SYCL_DAG'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
