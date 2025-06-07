import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import clone_detection_with_checkpoints

from codeaug.apply_stategies import Randomly
from codeaug.rename_strategies import (
    EnglishSynonyms,
    TNumber,
    TransformerMaskReplacement,
)
from codeaug.transforms import (
    Compose,
    InvertIfs,
    MoveRandomStmt,
    RemoveComments,
    RenameClasses,
    RenameFunctions,
    RenameVariables,
    ReplaceForWithWhile,
    SwitchConditionals,
    OneFrom,
    FormatWithAutoPEP8,
    FormatWithBlack,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base-mlm")
model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(device)
codebert_mask_replacement = TransformerMaskReplacement(tokenizer, model)
english_synonyms = EnglishSynonyms()
tnumber = TNumber()
transforms = [
    Compose(
        [
            OneFrom(
                [
                    FormatWithBlack(),
                    FormatWithAutoPEP8(aggressive=True),
                    FormatWithAutoPEP8(aggressive=False),
                ]
            ),
            RemoveComments(Randomly()),
            InvertIfs(Randomly()),
            ReplaceForWithWhile(Randomly()),
            SwitchConditionals(Randomly()),
            MoveRandomStmt(),
            MoveRandomStmt(),
            MoveRandomStmt(),
            RenameClasses(tnumber, Randomly(0.3)),
            RenameFunctions(tnumber, Randomly(0.3)),
            RenameVariables(tnumber, Randomly(0.3)),
            RenameClasses(english_synonyms, Randomly(0.3)),
            RenameFunctions(english_synonyms, Randomly(0.3)),
            RenameVariables(english_synonyms, Randomly(0.3)),
            RenameClasses(codebert_mask_replacement, Randomly(0.3)),
            RenameFunctions(codebert_mask_replacement, Randomly(0.3)),
            RenameVariables(codebert_mask_replacement, Randomly(0.3)),
        ]
    ),
]

from_chunk, to_chunk = [int(x) for x in input().split()]

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
clone_detection_with_checkpoints.get_clone_detection_dataloaders(
    tokenizer, from_chunk, to_chunk, transforms=transforms
)
