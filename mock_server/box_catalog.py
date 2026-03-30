import random


class BoxCatalog:
    BOX_POOL = [
        (0.600, 0.400, 0.400, 10.0),
        (0.500, 0.400, 0.300, 8.0),
        (0.400, 0.300, 0.200, 4.0),
        (0.350, 0.280, 0.190, 3.5),
        (0.406, 0.310, 0.211, 3.1),
        (0.300, 0.200, 0.150, 2.0),
        (0.250, 0.200, 0.150, 1.5),
        (0.200, 0.150, 0.100, 1.0),
        (0.450, 0.350, 0.250, 6.0),
        (0.550, 0.400, 0.300, 9.0),
        (0.380, 0.280, 0.200, 4.5),
        (0.320, 0.240, 0.180, 3.0),
    ]
    GAME_LENGTH = 120

    def __init__(self, seed: int = 42):
        self._seed = seed

    def generate(self, game_seed: int) -> list[dict]:
        rng = random.Random(self._seed ^ game_seed)
        boxes: list[dict] = []
        for _ in range(self.GAME_LENGTH):
            lx, ly, lz, w = rng.choice(self.BOX_POOL)
            lx = round(lx * rng.uniform(0.9, 1.1), 3)
            ly = round(ly * rng.uniform(0.9, 1.1), 3)
            lz = round(lz * rng.uniform(0.9, 1.1), 3)
            w = round(w * rng.uniform(0.9, 1.1), 2)
            box_id = str(rng.randint(100000, 999999))
            boxes.append({"id": box_id, "dimensions": [lx, ly, lz], "weight": w})
        return boxes
