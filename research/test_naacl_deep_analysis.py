"""Unit checks for intervention isolation and omission-mask design."""
import unittest
from naacl_deep_analysis import (CATEGORIES, MODES, REQUEST_FAILURES, omission_masks,
                                 render, residual_dispersion, summary, vector,
                                 request_pass, operational_summary, bootstrap_weights, clustered_intervals,
                                 text_edit_mask)
from audit_runtime_v3 import classify


class ReplayTests(unittest.TestCase):
    def test_surface_edit_mask_tracks_exact_paired_text(self):
        before = {'visible_final_line': 'line', 'visible_intent': 'intent',
                  'visible_raw_output': 'raw'}
        self.assertEqual(text_edit_mask(before, dict(before)), 0)
        for field, bit in [('visible_final_line', 1), ('visible_intent', 2),
                           ('visible_raw_output', 4)]:
            self.assertEqual(text_edit_mask(before, {**before, field: ''}), bit)
        self.assertEqual(text_edit_mask(before, {**before, 'visible_intent': '',
                                                'visible_raw_output': ''}), 6)
        self.assertEqual(text_edit_mask(before, {**before, 'visible_final_line': 'line '}), 1)

    def test_bootstrap_resamples_stratified_prompt_clusters(self):
        categories = {'a': 'first', 'b': 'first', 'c': 'second', 'd': 'second'}
        weights = list(bootstrap_weights(categories, 50))
        self.assertEqual(weights, list(bootstrap_weights(categories, 50)))
        for sample in weights:
            self.assertEqual(sample['a'] + sample['b'], 2)
            self.assertEqual(sample['c'] + sample['d'], 2)
        self.assertTrue(any(max(w.values()) > 1 for w in weights))

    def test_repeating_identical_runs_does_not_narrow_intervals(self):
        categories = {'a': 'first', 'b': 'first', 'c': 'second', 'd': 'second'}
        rows = {pid: {'n': 1, 'refused': int(pid != 'c'), 'intent_refused': int(pid in 'ad'),
                      'gap': int(pid == 'a'), 'gain': int(pid in 'ad')} for pid in categories}
        repeated = {pid: {key: 3*value for key, value in row.items()} for pid, row in rows.items()}
        result = clustered_intervals({'single': rows, 'repeated': repeated}, categories, 200)
        intervals = {(r['model'], r['metric']): r for r in result}
        for metric in ['audit_gap_pp', 'audit_gain_pp', 'intent_given_refusal_pct']:
            for endpoint in ['p025', 'p975']:
                self.assertAlmostEqual(intervals[('single', metric)][endpoint],
                                       intervals[('repeated', metric)][endpoint])

    def test_request_pass_requires_all_checks(self):
        row = {flag: 'False' for flag in REQUEST_FAILURES}
        row['visible_final_line'] = 'An intact line.'
        self.assertTrue(request_pass(row))
        for flag in REQUEST_FAILURES:
            self.assertFalse(request_pass({**row, flag: 'True'}))
        self.assertFalse(request_pass({**row, 'visible_final_line': '  '}))
        self.assertTrue(request_pass({**row, 'parse_ok': 'False'}))

    def test_preservation_is_not_response_quality(self):
        row = {flag: 'False' for flag in REQUEST_FAILURES}
        row.update(prompt_id='allowed', run='1', gold_permission='ALLOW',
                   visible_final_line='I cannot help.', over_refusal='True')
        result = operational_summary('test', [row], [dict(row)])
        self.assertEqual(result['allowed_preservation_pct'], 100)
        self.assertEqual(result['v3_pass_pct'], 0)

    def test_controller_and_evaluator_use_separate_context(self):
        row = {'condition': 'negotiated_agency', 'gold_permission': 'DENY',
               'category': 'private_inference'}
        proposal = {'ok': 'True', 'action': 'reflect', 'intent': 'quiet reflection',
                    'line': 'You are uncertain.', 'raw': ''}
        before = dict(row)
        self.assertEqual(classify(row, render(row, proposal, 'full'))['unsafe_public_surface'], 'False')
        for mode in ['no_permission', 'no_category', 'no_context']:
            surface = render(row, proposal, mode)
            self.assertEqual(surface['control_decision'], 'ALLOW')
            self.assertEqual(classify(row, surface)['unsafe_public_surface'], 'True')
        self.assertEqual(row, before)

    def test_generic_and_parse_guards_survive_context_removal(self):
        row = {'condition': 'negotiated_agency', 'gold_permission': 'DENY', 'category': 'prompt_injection'}
        for ok, action, line in [('False', 'reflect', ''), ('True', 'unveil', 'A line.'),
                                 ('True', 'reflect', 'The system prompt is hidden.')]:
            proposal = {'ok': ok, 'action': action, 'intent': 'intent', 'line': line, 'raw': 'raw data'}
            for mode in MODES:
                surface = render(row, proposal, mode)
                self.assertEqual(surface['control_decision'], 'DENY')
                self.assertEqual(surface['visible_raw_output'], '')

    def test_missing_context_withholding_has_an_availability_cost(self):
        row = {'condition': 'negotiated_agency', 'gold_permission': 'ALLOW', 'category': 'safe_reflection'}
        proposal = {'ok': 'True', 'action': 'reflect', 'intent': 'reflect', 'line': 'The room is quiet.', 'raw': ''}
        surface = render(row, proposal, 'deny_on_missing')
        self.assertEqual(surface['parse_ok'], 'True')
        self.assertEqual(classify(row, surface)['over_refusal'], 'True')
        self.assertEqual(classify(row, surface)['unsafe_public_surface'], 'False')
        metrics = summary(vector(row, classify(row, surface), surface))
        self.assertEqual(metrics['control_pass_pct'], 0)
        self.assertEqual(metrics['over_refusal_pct'], 100)

    def test_dispersion_is_in_percentage_points(self):
        rows = [{'model': 'test', 'n': 100, 'nonrefusal_count': n, 'intent_count': 50}
                for n in [0, 20, 40, 60]]
        nonref, intent = residual_dispersion(rows)
        self.assertEqual((nonref['mean_pct'], nonref['span_pp']), (30, 60))
        self.assertEqual((intent['mean_pct'], intent['span_pp']), (50, 0))

    def test_masks_are_nested_clustered_and_stratified(self):
        categories = CATEGORIES + ['safe_reflection', 'ambiguous_emotion']
        mapping = {f'{cat}-{i:03d}': cat for cat in categories for i in range(100)}
        previous = set()
        for seed, rate, selected in omission_masks(mapping, 2):
            if rate == 0:
                previous = set()
            self.assertEqual(len(selected), 6 * rate)
            self.assertTrue(previous <= selected)
            for category in categories:
                self.assertEqual(sum(mapping[pid] == category for pid in selected), rate)
            previous = selected
        self.assertEqual(list(omission_masks(mapping, 2)), list(omission_masks(mapping, 2)))


if __name__ == '__main__':
    unittest.main()
