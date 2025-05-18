import codeaug
import codebert_clone_detection


transform = codeaug.transforms.Compose([
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
])


codebert_clone_detection.train_model(t=transform)
