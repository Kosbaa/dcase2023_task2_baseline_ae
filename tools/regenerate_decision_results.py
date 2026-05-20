import argparse
import csv
from pathlib import Path

import numpy as np


def _read_anomaly_scores(csv_path: Path):
    rows = []
    scores = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0].strip()
            score_text = row[1].strip()
            if not filename:
                continue
            try:
                score = float(score_text)
            except ValueError:
                # Allow an optional header like: filename,score
                continue
            rows.append((filename, score))
            scores.append(score)

    if not rows:
        raise ValueError(f"No anomaly scores found in: {csv_path}")
    return rows, np.asarray(scores, dtype=np.float64)


def _write_decision_result(output_path: Path, rows, threshold: float):
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        for filename, score in rows:
            decision = 1 if score > threshold else 0
            writer.writerow([filename, decision])


def _format_k_name(k: float):
    if float(k).is_integer():
        return str(int(k))
    return str(k).replace(".", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate decision_result CSV files from anomaly_score CSV using fixed threshold sets."
    )
    parser.add_argument(
        "anomaly_score_csv",
        type=str,
        help="Path to anomaly_score CSV with rows: filename,score",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for decision_result CSV files (default: same directory as anomaly_score CSV).",
    )
    args = parser.parse_args()

    input_path = Path(args.anomaly_score_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, scores = _read_anomaly_scores(input_path)

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))

    percentile_modes = [95, 90, 85, 80]
    mean_plus_std_modes = [1.0, 1.5, 2.0]

    threshold_specs = []
    for p in percentile_modes:
        threshold_specs.append(
            (
                f"percentile{p}",
                float(np.percentile(scores, p)),
                output_dir / f"decision_result_percentile{p}.csv",
            )
        )
    for k in mean_plus_std_modes:
        threshold_specs.append(
            (
                f"mean_plus_{k}std",
                float(score_mean + k * score_std),
                output_dir / f"decision_result_mean_plus_{_format_k_name(k)}std.csv",
            )
        )

    for mode_name, threshold, output_path in threshold_specs:
        _write_decision_result(output_path=output_path, rows=rows, threshold=threshold)
        predicted_anomalies = int(np.sum(scores > threshold))
        print(f"[{mode_name}]")
        print(f"  threshold: {threshold:.10f}")
        print(f"  predicted_anomalies: {predicted_anomalies}")
        print(f"  min_score: {score_min:.10f}")
        print(f"  max_score: {score_max:.10f}")
        print(f"  mean_score: {score_mean:.10f}")
        print(f"  std_score: {score_std:.10f}")
        print(f"  output_csv: {output_path}")


if __name__ == "__main__":
    main()
