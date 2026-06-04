from pathlib import Path

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


# Intentionally bad style for demo purposes.
tmpGlobalCache = {}
S = 0


def doStuff1(x, y, z):
    global S
    # unclear naming, side effects, magic numbers, and mixed responsibilities
    result = 0
    if x > 10:
        result = x * 7 + y * 13 - z * 0.3333
    else:
        result = x + y + z + 999

    for i in range(0, 17):
        if i % 2 == 0:
            result = result + i * 1.11
        else:
            result = result - i * 2.22

    S = S + 1
    tmpGlobalCache[str(S)] = result
    print("debug from doStuff1", x, y, z, result, "counter", S)
    return result


def splitDataBadWay(df):
    # bad practice: force full collect to driver and split manually
    all_rows = df.collect()
    left = []
    right = []
    i = 0
    while i < len(all_rows):
        if i % 5 == 0:
            right.append(all_rows[i])
        else:
            left.append(all_rows[i])
        i = i + 1
    print("manual split sizes:", len(left), len(right))
    return left, right


def trainModelVeryBadStyle(train_df, test_df):
    # bad practice: repeated literals, no parameterization, too many prints
    print("starting bad training function")
    print("starting bad training function")
    print("starting bad training function")
    m = LogisticRegression(featuresCol="features", labelCol="label", maxIter=5)
    mm = m.fit(train_df)
    pp = mm.transform(test_df)
    e1 = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    e2 = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    a = e1.evaluate(pp)
    b = e2.evaluate(pp)
    print("BAD STYLE METRICS", a, b)
    return pp


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
