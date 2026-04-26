class ELOTracker:
    def __init__(self, k=32):
        self.ratings = {}
        self.k = k
        self.history = []

    def get_rating(self, agent: str) -> float:
        return self.ratings.get(agent, 1000.0)

    def update(self, winner: str, loser: str) -> dict:
        ra = self.get_rating(winner)
        rb = self.get_rating(loser)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea
        self.ratings[winner] = ra + self.k * (1 - ea)
        self.ratings[loser] = rb + self.k * (0 - eb)
        entry = {
            "winner": winner,
            "loser": loser,
            "new_winner_elo": round(self.ratings[winner], 1),
            "new_loser_elo": round(self.ratings[loser], 1),
            "round": len(self.history) + 1
        }
        self.history.append(entry)
        return entry

    def get_standings(self) -> dict:
        return {
            "generator_elo": round(self.get_rating("generator"), 1),
            "detector_elo": round(self.get_rating("detector"), 1),
            "total_rounds": len(self.history),
            "current_leader": max(
                self.ratings, key=self.ratings.get
            ) if self.ratings else "none"
        }
