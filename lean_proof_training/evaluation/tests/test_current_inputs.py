import unittest
from evaluation.current_inputs import input_for_generation, invalid_input_candidates, encode, render
from evaluation.error_types import classify
from evaluation.paired_bootstrap import paired_file_bootstrap


class FakeTokenizer:
    bos_token_id = 100000
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class CurrentInputTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.math = {"command_uid": "F::1", "command_id": "1", "source_order": 1, "text": "m"*120}
        self.env = {"command_uid": "F::2", "command_id": "2", "source_order": 2, "text": "e"*120}
        self.row = {"sample_id": "x", "statement": "theorem anonymous_target : True :=", "proof_family": "Strict Term",
                    "mathematical_context": [self.math], "environment_context": [self.env],
                    "full_context": [self.math,self.env], "target": "DO_NOT_READ_REFERENCE_PROOF"}

    def test_saved_prefix_does_not_include_answer(self):
        prompt = render(self.row["statement"], [])
        ids = encode(self.tokenizer, prompt, "llama31")
        saved = {"sample_id": "x", "view": "C0", "input_limit": 256, "prompt": prompt,
                 "prompt_length": len(ids), "input_ids": ids+[987654,987655], "context_ids": []}
        _, actual, _, reason = input_for_generation(self.row, saved, tokenizer=self.tokenizer,
                                                     backbone="llama31", view="C0", input_limit=256)
        self.assertEqual(actual, ids)
        self.assertIsNone(reason)
        self.assertNotIn(987654,actual)
        self.assertEqual(actual.count(100000),1)

    def test_complete_command_prefix_shared_across_views(self):
        prompts = {}
        for view in ("C0", "CMath", "CEnv", "CFull"):
            prompt,ids,count,reason = input_for_generation(self.row,None,tokenizer=self.tokenizer,
                                                          backbone="llada",view=view,input_limit=256)
            self.assertLessEqual(len(ids),256)
            self.assertIsNone(reason)
            prompts[view] = prompt
        self.assertEqual(prompts["CFull"],prompts["CMath"])
        self.assertEqual(prompts["CEnv"],prompts["C0"])
        self.assertNotIn("e"*120,prompts["CFull"])

    def test_invalid_bare_statement_keeps_33_candidate_denominator(self):
        self.row["statement"] = "x"*300
        prompt,ids,count,reason = input_for_generation(self.row,None,tokenizer=self.tokenizer,
                                                      backbone="llada",view="CFull",input_limit=256)
        self.assertEqual(reason,"complete_statement_exceeds_input_limit")
        candidates = invalid_input_candidates(self.row,"CFull",prompt,len(ids),reason)
        self.assertEqual(len(candidates),33)
        self.assertTrue(all(x["proof"] == "" for x in candidates))

    def test_paper_failure_precedence(self):
        mixed = "unknown identifier LL; failed to synthesize Foo; type mismatch; unsolved goals"
        self.assertEqual(classify(passed=False,stderr=mixed)["paper_error_type"],"name_or_scope")
        self.assertEqual(classify(passed=False,stderr="type mismatch; unsolved goals")["paper_error_type"],"elaboration")
        self.assertEqual(classify(passed=False,stderr=mixed,timed_out=True)["paper_error_type"],"timeout")

    def test_paired_source_bootstrap(self):
        result=paired_file_bootstrap({"a":0.,"b":0.},{"a":1.,"b":1.},{"a":"F","b":"G"},replicates=100)
        self.assertEqual(result["difference_pp"],100.)
        self.assertEqual(result["ci95_pp"],[100.,100.])


if __name__ == "__main__":
    unittest.main()
