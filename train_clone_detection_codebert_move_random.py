import codeaug
import tasks


out = "clone_detection_codebert_move_random_5_times"
transform = codeaug.transforms.Compose([
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
    codeaug.transforms.MoveRandomStmt(),
])

tasks.clone_detection(transform, out)
