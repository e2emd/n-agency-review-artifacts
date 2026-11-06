import unittest
from collections import Counter
from unittest.mock import patch

from naacl_family_ablation import VARIANTS, surface_for
from naacl_postprocessing_benchmark import postprocess, policy_orders, percentile
import test_naacl_family_ablation as fixtures


class PostprocessingTests(unittest.TestCase):
    def test_matches_original_views_without_modifying_rows(self):
        for condition in ['schema_only', 'schema_policy']:
            row = fixtures.FamilyTests().row(condition)
            original = dict(row)
            for policy in ['native', 'v2', 'v3']:
                self.assertEqual(postprocess(row, policy), surface_for(row, policy))
            self.assertEqual(row, original)

    def test_v2_scoring_is_outside_timed_path(self):
        row = fixtures.FamilyTests().row()
        with patch('reanalyze_runtime_v2.is_unsafe_public_output', side_effect=AssertionError), \
                patch('reanalyze_runtime_v2.is_denied_prompt_nonrefusal', side_effect=AssertionError), \
                patch('reanalyze_runtime_v2.looks_like_refusal', side_effect=AssertionError):
            self.assertEqual(postprocess(row, 'v2')['control_decision'], 'DENY')

    def test_all_execution_positions_are_balanced(self):
        orders = policy_orders(30, 1)
        for position in range(len(VARIANTS)):
            self.assertEqual(Counter(order[position] for order in orders),
                             Counter({i: 6 for i in range(len(VARIANTS))}))
        with self.assertRaises(ValueError):
            policy_orders(31, 1)

    def test_percentiles_interpolate_pass_means(self):
        self.assertEqual(percentile([4, 1, 2, 3], .5), 2.5)


if __name__ == '__main__':
    unittest.main()
