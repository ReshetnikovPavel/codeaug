import codebert_clone_detection
from codeaug.transforms import Compose, RenameClasses, RenameFunctions, RenameVariables

transform = Compose([
    RenameClasses(),
    RenameFunctions(),
    RenameVariables()
])


codebert_clone_detection.train_model(t=transform)
