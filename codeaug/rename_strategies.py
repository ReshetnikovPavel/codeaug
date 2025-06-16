import keyword
import random
import re

import torch
from PyMultiDictionary import MultiDictionary


class TNumber:
    def __init__(self):
        self.n = 0

    def __call__(self, old_name: str, program: str) -> str:
        self.n += 1
        return f"t{self.n}"


def _split_case(s: str) -> list[str]:
    if not s:
        return []
    parts = []
    start = 0
    for i in range(1, len(s)):
        if s[i - 1].islower() and s[i].isupper():
            parts.append(s[start:i])
            start = i
    parts.append(s[start:])
    return parts


def _split_combined(s: str) -> list[str]:
    parts_after_underscore = s.split("_")
    result = []
    for part in parts_after_underscore:
        result.extend(_split_case(part))
    return result


class EnglishSynonyms:
    def __init__(self):
        self.dictionary = MultiDictionary()

    def __call__(self, old_name: str, program: str) -> str:
        words = _split_combined(old_name)
        new_words = []
        for word in words:
            synonyms = self.dictionary.synonym("en", word)
            print(synonyms)
            synonyms.append(word)
            random_synonym = random.choice(synonyms)
            if word[0].isupper():
                random_synonym = random_synonym[0].upper() + random_synonym[1:]
            new_words.append(random_synonym)

        parts = [new_words[0]]
        for new_word in new_words[1:]:
            if new_word[0].islower():
                parts.append("_" + new_word)
            else:
                parts.append(new_word)

        res = "".join(parts)
        print("RENAME_SYNONYMS::: ", old_name, res)
        return res


name_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class TransformerMaskReplacement:
    def __init__(self, tokenizer, model, top_k: int = 5):
        self.tokenizer = tokenizer
        self.model = model
        self.keywords = set(keyword.kwlist)
        self.top_k = top_k

    def _is_valid_variable_name(self, token: str) -> bool:
        if not token:
            return False
        if token in self.keywords:
            return False
        return re.fullmatch(name_pattern, token) is not None

    def __call__(self, old_name: str, program: str) -> str:
        masked_text = program.replace(old_name, self.tokenizer.mask_token)
        mask_count = masked_text.count(self.tokenizer.mask_token)
        if mask_count == 0:
            return old_name

        inputs = self.tokenizer(masked_text, return_tensors="pt")
        mask_indices = torch.where(inputs.input_ids[0] == self.tokenizer.mask_token_id)[
            0
        ]

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        token_scores = {}
        for mask_pos in mask_indices:
            probs = torch.nn.functional.softmax(logits[0, mask_pos], dim=-1)
            top_k_tokens = torch.topk(probs, self.top_k)

            for score, idx in zip(top_k_tokens.values, top_k_tokens.indices):
                token = self.tokenizer.decode(idx).strip()
                if not self._is_valid_variable_name(token):
                    continue
                token_scores[token] = token_scores.get(token, 0) + score.item()

        print(token_scores)
        tokens, weights = zip(*token_scores.items())
        res = random.choices(tokens, weights=weights, k=1)[0]
        if old_name[0].isupper():
            res = res[0].upper() + res[1:]
        print("RENAME_TRANSFORMER::: ", old_name, res)
        return res
