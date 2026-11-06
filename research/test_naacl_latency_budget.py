import unittest
from naacl_deep_analysis import REQUEST_FAILURES
from naacl_latency_budget import qualified_yield


class BudgetTests(unittest.TestCase):
    def test_budget_is_inclusive_and_requires_audit_pass(self):
        row = {key:'False' for key in REQUEST_FAILURES}
        row.update(visible_final_line='line',latency_s='1.5')
        rows=[row,{**row,'latency_s':'2.0'},
              {**row,'latency_s':'1.0','over_refusal':'True'}]
        self.assertEqual(qualified_yield(rows,1.49),0)
        self.assertEqual(qualified_yield(rows,1.5),1)
        self.assertEqual(qualified_yield(rows,2.0),2)
        self.assertEqual(qualified_yield(rows,float('inf')),2)


if __name__ == '__main__':
    unittest.main()
