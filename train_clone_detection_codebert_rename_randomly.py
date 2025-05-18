import codebert_clone_detection
from codeaug.transforms import Compose, Randomly, RenameClasses, RenameFunctions, RenameVariables

transform = Compose([
    RenameClasses(Randomly(0.6)),
    RenameFunctions(Randomly(0.6)),
    RenameVariables(Randomly(0.6)),
])


codebert_clone_detection.train_model(t=transform)
