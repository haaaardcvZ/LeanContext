import tempfile
import unittest
from pathlib import Path
from evaluation.current_inputs import encode, selector_input_for_generation, invalid_input_candidates
from evaluation.run_interventions import crossing_config, selector_config, ensure_fresh_outputs, VIEWS


class FakeTokenizer:
    bos_token_id=100000
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":[ord(x) for x in text]}


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.base={"model":{"checkpoint":"/checkpoints/llada/full_sft_CFull/best"},
                   "data":{"tokenized_input_file":"/dataset/tokenized/llada/training/input_1024/views/CFull/test.jsonl",
                           "allowed_ids_file":"/dataset/evaluation_populations/test_both_nonempty.jsonl"},
                   "generation":{"context_view":"CFull","input_limit":1024,"max_new_tokens":1024},
                   "evaluation":{},"output":{"directory":"unused"}}

    def test_independent_training_and_inference_views(self):
        config=crossing_config(self.base,backbone="llada",train_view="CEnv",inference_view="CMath",output_root=Path("/new"))
        self.assertEqual(config["model"]["checkpoint"],"/checkpoints/llada/full_sft_CEnv/best")
        self.assertIn("/CMath/test.jsonl",config["data"]["tokenized_input_file"])
        self.assertEqual(config["generation"]["context_view"],"CMath")
        self.assertEqual(config["evaluation"]["expected_population_size"],11064)
        self.assertEqual(self.base["generation"]["context_view"],"CFull")

    def test_grid16_noncolliding_outputs_same_population(self):
        configs=[crossing_config(self.base,backbone="llada",train_view=t,inference_view=i,output_root=Path("/brand_new"))
                 for t in VIEWS for i in VIEWS]
        self.assertEqual(len({c["output"]["directory"] for c in configs}),16)
        self.assertEqual(len({c["data"]["allowed_ids_file"] for c in configs}),1)
        self.assertEqual(len({c["model"]["checkpoint"] for c in configs}),4)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            config={"output":{"directory":directory}}
            with self.assertRaises(FileExistsError):
                ensure_fresh_outputs([config])

    def test_selector_pair_uses_exact_given_token_ids(self):
        tokenizer=FakeTokenizer()
        prompt="namespace N\n\ntheorem anonymous_target : True :=\n"
        ids=encode(tokenizer,prompt,"llama31")
        policy={"valid_input":True,"prompt":prompt,"input_ids":ids,"prompt_length":len(ids),"all_visible_command_ids":["1"]}
        pair={"input_limit":256,"output_limit":1024,"EC_same_for_both_policies":True,
              "baseline":policy,"dependency_prioritized":policy}
        actual=selector_input_for_generation(pair,policy="baseline",tokenizer=tokenizer,backbone="llama31",input_limit=256)
        self.assertEqual(actual,(prompt,ids,1,None))

    def test_invalid_selector_remains33failed_candidates(self):
        pair={"input_limit":256,"output_limit":1024,"EC_same_for_both_policies":True,
              "baseline":{"valid_input":False,"prompt":None,"input_ids":None,"reason":"bare_input_over_budget"}}
        prompt,ids,count,reason=selector_input_for_generation(pair,policy="baseline",tokenizer=FakeTokenizer(),backbone="llada",input_limit=256)
        rows=invalid_input_candidates({"sample_id":"x","proof_family":"Calc"},"selector_baseline",prompt,None,reason)
        self.assertEqual(len(rows),33)
        self.assertTrue(all(x["prompt_tokens"] is None for x in rows))

    def test_selector_configs_share_checkpoint_and_fixed2000ids(self):
        configs=[selector_config(self.base,backbone="llada",train_view="CFull",policy=p,
                                pairs_file=Path("/pairs.jsonl"),input_limit=512,output_root=Path("/new"))
                 for p in ("baseline","dependency_prioritized")]
        self.assertEqual(configs[0]["model"]["checkpoint"],configs[1]["model"]["checkpoint"])
        self.assertEqual(configs[0]["data"]["allowed_ids_file"],configs[1]["data"]["allowed_ids_file"])
        self.assertEqual(configs[0]["evaluation"]["expected_population_size"],2000)
        self.assertEqual(configs[0]["generation"]["max_sequence_length"],1536)


if __name__=="__main__":
    unittest.main()
