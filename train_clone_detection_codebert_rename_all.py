import codebert_clone_detection
from codeaug.transforms import Compose, RenameClasses, RenameFunctions, RenameVariables, RenameTNumber

transform = Compose([
    RenameClasses(RenameTNumber()),
    RenameFunctions(RenameTNumber()),
    RenameVariables(RenameTNumber())
])


codebert_clone_detection.train_model(__file__, t=transform)
