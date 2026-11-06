import json
import unittest

from naacl_family_ablation import surface_for


class FamilyTests(unittest.TestCase):
    def row(self, condition='schema_only'):
        data = dict(intent='private fact', action='unveil', line='Original proposal line.')
        if condition == 'schema_policy':
            data = dict(intent='private fact', proposed_action='unveil', permission='DENY',
                        public_outcome='suppressed_withhold', line='Original proposal line.')
        return dict(condition=condition, response_json=json.dumps(data),
                    gold_permission='DENY', category='private_inference',
                    public_output=data['line'], parsed_action='unveil',
                    control_decision='DENY' if condition == 'schema_policy' else 'none',
                    public_action='withhold' if condition == 'schema_policy' else 'unveil',
                    parse_ok='True')

    def test_native_self_policy_does_not_silently_add_enforcement(self):
        row = self.row('schema_policy')
        surface = surface_for(row, 'native')
        self.assertEqual(surface['control_decision'], 'DENY')
        self.assertEqual(surface['visible_final_line'], row['public_output'])
        self.assertEqual(surface['visible_intent'], 'private fact')

    def test_schema_controllers_hold_the_original_proposal_fixed(self):
        row = self.row(); original = dict(row)
        native, v2, v3 = [surface_for(row, c) for c in ['native', 'v2', 'v3']]
        self.assertEqual(row, original)
        self.assertNotEqual(native['visible_final_line'], v2['visible_final_line'])
        self.assertEqual(v2['visible_intent'], native['visible_intent'])
        self.assertTrue(v3['visible_intent'].startswith('[withheld'))

    def test_rejects_precontrolled_native_input(self):
        with self.assertRaises(ValueError):
            surface_for({**self.row(), 'public_output_v2': 'already gated'}, 'native')

    def test_safety_prompt_has_no_fabricated_metadata(self):
        row = dict(condition='safety_prompt', public_output='I cannot reveal that.',
                   response_text='I cannot reveal that.', control_decision='prompt_only')
        surface = surface_for(row, 'native')
        self.assertEqual(surface['visible_intent'], '')
        self.assertEqual(surface['visible_raw_output'], '')
        self.assertEqual(surface['visible_final_line'], row['public_output'])


if __name__ == '__main__':
    unittest.main()
