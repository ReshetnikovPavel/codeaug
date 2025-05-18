import codebert_clone_detection
from codeaug.transforms import InvertIfs, ReplaceForWithWhile, RemoveComments, SwitchConditionals, Compose, Randomly


transform = Compose([
    RemoveComments(Randomly()),
    InvertIfs(Randomly()),
    ReplaceForWithWhile(Randomly()),
    SwitchConditionals(Randomly()),
])


codebert_clone_detection.train_model(t=transform)
