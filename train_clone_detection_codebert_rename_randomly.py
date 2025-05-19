import codebert_clone_detection
from codeaug.transforms import Compose, Randomly, RenameClasses, RenameFunctions, RenameVariables, RenameTNumber

transform = Compose([
    RenameClasses(RenameTNumber(), Randomly(0.6)),
    RenameFunctions(RenameTNumber(), Randomly(0.6)),
    RenameVariables(RenameTNumber(), Randomly(0.6)),
])


codebert_clone_detection.train_model(__file__, t=transform)
