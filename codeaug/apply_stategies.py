import codeaug.utils


class Randomly:
    def __init__(self, probability: float = 0.8):
        self.probability = probability

    def __call__(self, *args) -> bool:
        return codeaug.utils.randomly(self.probability)
