import codeaug
import codebert_clone_detection


transform = codeaug.transforms.Compose([
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
])


codebert_clone_detection.train_model(__file__, t=transform)
