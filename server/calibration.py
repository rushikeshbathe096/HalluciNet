class CalibrationTracker:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.records = []

    def record(self, confidence: float, was_correct: bool):
        self.records.append((confidence, was_correct))

    def get_calibration_curve(self) -> dict:
        if not self.records:
            return {
                "bins": [],
                "calibration_error": 0.0,
                "total_samples": 0,
                "interpretation": "No data yet"
            }
        bins = [[] for _ in range(self.n_bins)]
        for conf, correct in self.records:
            bin_idx = min(int(conf * self.n_bins), self.n_bins - 1)
            bins[bin_idx].append(correct)
        curve = []
        for i, b in enumerate(bins):
            if b:
                bin_conf = (i + 0.5) / self.n_bins
                bin_acc = sum(b) / len(b)
                curve.append({
                    "confidence_bin": round(bin_conf, 2),
                    "actual_accuracy": round(bin_acc, 2),
                    "sample_count": len(b),
                    "calibration_gap": round(abs(bin_conf - bin_acc), 3)
                })
        total = len(self.records)
        ece = sum(
            (len(b) / total) * abs(
                (i + 0.5) / self.n_bins - (sum(b) / len(b))
            )
            for i, b in enumerate(bins) if b
        )
        return {
            "bins": curve,
            "calibration_error": round(ece, 4),
            "total_samples": total,
            "interpretation": (
                "Well calibrated" if ece < 0.1 else
                "Overconfident" if ece < 0.2 else
                "Poorly calibrated"
            )
        }
