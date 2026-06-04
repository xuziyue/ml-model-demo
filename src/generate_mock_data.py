import csv
import math
import random
from pathlib import Path


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def build_row() -> dict:
    age = random.randint(18, 70)
    income = round(random.gauss(70000, 18000), 2)
    income = max(18000.0, min(income, 130000.0))

    monthly_spend = round(random.gauss(1100, 500), 2)
    monthly_spend = max(100.0, min(monthly_spend, 3000.0))

    tenure_months = random.randint(1, 120)

    score = (
        -4.0
        + 0.05 * age
        + 0.00003 * income
        - 0.001 * monthly_spend
        + 0.015 * tenure_months
    )
    probability = sigmoid(score)
    label = 1 if random.random() < probability else 0

    return {
        "age": age,
        "income": income,
        "monthly_spend": monthly_spend,
        "tenure_months": tenure_months,
        "label": label,
    }


def generate_mock_data(output_path: Path, row_count: int = 500, seed: int = 42) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["age", "income", "monthly_spend", "tenure_months", "label"]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(row_count):
            writer.writerow(build_row())


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    target_file = project_root / "data" / "mock_customer_data.csv"
    generate_mock_data(target_file)
    print(f"Mock data written to: {target_file}")
