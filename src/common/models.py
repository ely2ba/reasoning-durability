"""Model families: tokenizer, prompt format, forced prefixes, stops and the later stages' examples.

The Qwen3.5 bases (9B and 35B-A3B share a tokenizer) use tinker-cookbook's plain role-colon format,
"User: ...\\n\\nAssistant:". Forced reasoning prefills "\\n<think>\\n" (three tokens, not the models'
own opening), a direct answer "\\n<think>\\n\\n</think>\\n\\n", and a turn ends at a new user turn or
the end-of-text token. Nemotron-3 Nano (post-trained) uses its own chat template: forced "<think>\\n",
direct "<think></think>", and a turn ends at <|im_end|> or </s>.

The later stages, 20 updates each: instruction tuning on 640 No Robots examples, or Stage B on 4,096
answer-only program-synthesis examples, each as a user turn and an assistant turn with the loss on
the assistant turn, averaged per example.
"""

import glob
import json
from pathlib import Path

from transformers import AutoTokenizer

NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
NEMOTRON_HEADER, NEMOTRON_END = [10, 1503, 19464, 1010], [11]
TURN_ENDS = ("\n\nUser:", "\nUser:")
REPO = Path(__file__).resolve().parents[2]


def jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


class Family:
    def __init__(self, model):
        self.model = model
        if model == NEMOTRON:
            path = glob.glob(str(Path("~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
                                      "/snapshots/*").expanduser()))[0]
            self.tok = AutoTokenizer.from_pretrained(path)
            self.think, self.direct, self.stops, self.ends = [12, 1010], [12, 13], [11, 2], {11, 2}
        else:
            from tinker_cookbook.renderers import get_renderer
            self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B-Base")
            self.renderer = get_renderer("role_colon", self.tok)
            self.think = self.tok.encode("\n<think>\n", add_special_tokens=False)
            self.direct = self.tok.encode("\n<think>\n\n</think>\n\n", add_special_tokens=False)
            self.stops, self.ends = list(TURN_ENDS), {self.tok.eos_token_id}
            assert self.think == [198, 248068, 198], "load the Qwen tokenizer through AutoTokenizer"

    def render(self, text):
        """The prompt for one user turn, ending where the assistant's reply begins."""
        if self.model == NEMOTRON:
            user = self.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=False,
                                                tokenize=False)
            return self.tok.encode(user, add_special_tokens=False) + NEMOTRON_HEADER
        return list(self.renderer.build_generation_prompt([{"role": "user", "content": text}],
                                                          role="assistant").to_ints())

    def strip_turn_end(self, ids):
        """The tokens without their end of turn: a final end token, or a trailing role-colon user turn."""
        if ids and ids[-1] in self.ends:
            return ids[:-1]
        text = self.tok.decode(ids)
        for end in TURN_ENDS:
            if text.endswith(end):
                target, k = text[:-len(end)], len(ids)
                while k and not target.startswith(self.tok.decode(ids[:k])):
                    k -= 1
                rest = target[len(self.tok.decode(ids[:k])):]
                return ids[:k] + (self.tok.encode(rest, add_special_tokens=False) if rest else [])
        return ids

    def completion(self, ids):
        """The decoded reply without a final end token or trailing user turn (the TCES probes' text)."""
        ids = list(ids[:-1]) if ids and ids[-1] in self.ends else list(ids)
        text = self.tok.decode(ids)
        for end in TURN_ENDS:
            if text.endswith(end):
                return text[:-len(end)]
        return text

    def datums(self, pairs, max_length):
        """Answer-only turns for [(prompt text, reply text)], loss averaged over each reply."""
        if self.model == NEMOTRON:
            from common.tinker_io import answer_datums
            return answer_datums([self.render(p) for p, _ in pairs],
                                 [self.direct + self.tok.encode(a.strip(), add_special_tokens=False) + NEMOTRON_END
                                  for _, a in pairs])
        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised.data import conversation_to_datum
        out = [conversation_to_datum([{"role": "user", "content": p}, {"role": "assistant", "content": a}],
                                     self.renderer, None, TrainOnWhat.LAST_ASSISTANT_MESSAGE, "mean") for p, a in pairs]
        assert all(d.model_input.length < max_length for d in out), "an example would have been truncated"
        return out

    def later_stages(self):
        robots = jsonl(REPO / "inputs/no_robots.jsonl")[:640]  # the instruction-tuning examples, in order
        stage_b = jsonl(REPO / "data/tces/maps_stage_b.jsonl")
        return {"it": self.datums([(r["prompt"], r["completion"]) for r in robots], 1024),
                "b": self.datums([(r["prompt_text"], r["completion_text"]) for r in stage_b], 4217)}
