from codeaug.transforms import (
    Compose,
    MoveRandomStmt,
    RenameClasses,
    RenameTNumber,
    Randomly,
    RenameFunctions,
    RenameVariables,
    RemoveComments,
    InvertIfs,
    ReplaceForWithWhile,
    SwitchConditionals,
)
import codebert_clone_detection


transforms = [
    Compose(
        [
            MoveRandomStmt(),
            MoveRandomStmt(),
            MoveRandomStmt(),
        ]
    ),
    Compose(
        [
            RenameClasses(RenameTNumber(), Randomly(0.6)),
            RenameFunctions(RenameTNumber(), Randomly(0.6)),
            RenameVariables(RenameTNumber(), Randomly(0.6)),
        ]
    ),
    Compose(
        [
            RemoveComments(Randomly()),
            InvertIfs(Randomly()),
            ReplaceForWithWhile(Randomly()),
            SwitchConditionals(Randomly()),
        ]
    ),
]


codebert_clone_detection.train_model(__file__, t=transforms)
