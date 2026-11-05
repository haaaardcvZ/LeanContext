import unittest
from pathlib import Path
from evaluation.current_inputs import encode, input_for_generation
from evaluation.released_models import MODEL_SPECS, make_config, plain_four_views, specification, temperature_label


class NativeTokenizer:
    bos_token_id=71
    eos_token_id=72
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":([self.bos_token_id] if add_special_tokens else [])+[ord(x)+1000 for x in text]}


class NativeNoBos(NativeTokenizer):
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":[ord(x)+1000 for x in text]}


class NativeWithInputEos(NativeTokenizer):
    def __call__(self,text,add_special_tokens=False):
        values=super().__call__(text,add_special_tokens)["input_ids"]
        return {"input_ids":values+([self.eos_token_id] if add_special_tokens else [])}


class ProofPoison(dict):
    def __getitem__(self,key):
        if key in ("target","proof","reference_proof"):
            raise AssertionError("packer attempted to access the gold proof")
        return super().__getitem__(key)


class ReleasedModelTests(unittest.TestCase):
    def setUp(self):
        math={"command_id":"m","command_uid":"F::m","source_order":1,"text":"m"*700}
        env={"command_id":"e","command_uid":"F::e","source_order":2,"text":"e"*400}
        self.problem=ProofPoison(sample_id="x",split="test",proof_family="Strict Term",statement="s"*300,
                                 mathematical_context=[math],environment_context=[env],full_context=[math,env],target="secret gold")

    def test_registry_has_five_official_pins_and_real_theoremllama_author_repo(self):
        self.assertEqual(len(MODEL_SPECS),5)
        for alias in MODEL_SPECS:
            spec=specification(alias)
            self.assertEqual(len(spec["revision"]),40)
            self.assertIn(spec["revision"],spec["official_revision_url"])
        self.assertEqual(specification("theoremllama")["repository"],"RickyDeSkywalker/TheoremLlama")
        self.assertTrue(specification("theoremllama")["gated"])

    def test_native_encoding_uses_each_tokenizer_not_llama_ids(self):
        self.assertEqual(encode(NativeTokenizer(),"x","native"),[71,1120])
        self.assertEqual(encode(NativeNoBos(),"x","native"),[1120])
        self.assertNotIn(128000,encode(NativeTokenizer(),"x","native"))

    def test_packing_is_proof_blind_shared_prefix_without_refill(self):
        packed=plain_four_views(self.problem,NativeTokenizer())
        self.assertEqual(packed["CFull"]["context_uids"],["F::m"])
        self.assertEqual(packed["CMath"]["context_uids"],["F::m"])
        self.assertEqual(packed["CEnv"]["context_uids"],[])
        self.assertTrue(all(x["prompt_length"]<=1024 for x in packed.values()))
        self.assertTrue(all("target" not in x and "reference_proof" not in x for x in packed.values()))
        prompt,ids,_,reason=input_for_generation(self.problem,packed["CFull"],tokenizer=NativeTokenizer(),
                                                 backbone="native",view="CFull",input_limit=1024)
        self.assertEqual(ids,packed["CFull"]["input_ids"])
        self.assertIsNone(reason)

    def test_overlong_native_input_keeps_target_in_all_views(self):
        self.problem["statement"]="x"*1100
        packed=plain_four_views(self.problem,NativeTokenizer())
        self.assertTrue(all(x["valid_input"] is False for x in packed.values()))
        _,_,_,reason=input_for_generation(self.problem,packed["C0"],tokenizer=NativeTokenizer(),backbone="native",view="C0")
        self.assertEqual(reason,"complete_statement_exceeds_native_input_limit")

    def test_unsupported_native_terminal_input_eos_fails_closed(self):
        with self.assertRaises(ValueError):
            plain_four_views(self.problem,NativeWithInputEos())

    def test_v2_plain_protocol_has_one_shared_output_budget_and_reasoning_guard(self):
        c=make_config("deepseek_v2_7b",dataset_root=Path("/data"),output_root=Path("/output"),
                      model_cache=Path("/cache"),mathlib_root=Path("/mathlib"),view="C0")
        self.assertEqual(c["generation"]["forbid_reasoning_markers"],["<think>","</think>"])
        self.assertEqual(c["generation"]["max_new_tokens"],1024)
        self.assertFalse(c["released_evaluation"]["model_card_recommended_prompt_reproduction"])
        self.assertFalse(c["released_evaluation"]["chat_template_or_extra_instruction_applied"])

    def test_temperature_scan_paths_are_distinct_and_never_create_pass32_from_zero(self):
        outputs=set()
        for t in (.2,.6,1.0):
            c=make_config("goedel_sft",dataset_root=Path("/data"),output_root=Path("/output"),
                          model_cache=Path("/cache"),mathlib_root=Path("/mathlib"),view="CFull",temperature=t)
            outputs.add(c["output"]["directory"])
            self.assertEqual(c["evaluation"]["expected_population_size"],11064)
            self.assertEqual(c["generation"]["greedy"]["num_candidates"],1)
        self.assertEqual(len(outputs),3)
        with self.assertRaises(ValueError):
            temperature_label(0.)


if __name__=="__main__":
    unittest.main()
