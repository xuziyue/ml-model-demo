from pathlib import Path

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


def train_model(data_path: Path) -> None:
    spark = (
        SparkSession.builder.appName("pyspark-logistic-regression-demo")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        df = spark.read.csv(str(data_path), header=True, inferSchema=True)

        # Keep only the columns used by this demo and remove rows with missing values.
        df = df.select("age", "income", "monthly_spend", "tenure_months", "label").dropna()

        assembler = VectorAssembler(
            inputCols=["age", "income", "monthly_spend", "tenure_months"],
            outputCol="features",
        )
        model_input = assembler.transform(df).select("features", "label")

        train_df, test_df = model_input.randomSplit([0.8, 0.2], seed=42)

        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30)
        model = lr.fit(train_df)

        predictions = model.transform(test_df)

        roc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        accuracy_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

        auc = roc_eval.evaluate(predictions)
        accuracy = accuracy_eval.evaluate(predictions)

        print("=== Model Metrics ===")
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print()

        print("=== Model Parameters ===")
        print(f"Coefficients: {model.coefficients}")
        print(f"Intercept: {model.intercept}")
        print()

        print("=== Sample Predictions ===")
        predictions.select("label", "probability", "prediction").show(10, truncate=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "mock_customer_data.csv"
    train_model(csv_path)
