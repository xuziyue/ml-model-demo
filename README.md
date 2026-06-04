# PySpark Logistic Regression Demo

Simple demo project that:
- creates mock customer data as a local CSV file
- reads the CSV with PySpark
- splits data into training and testing sets
- trains a logistic regression model
- prints evaluation metrics and sample predictions

## Project Structure

- `data/mock_customer_data.csv`: local mock dataset
- `src/generate_mock_data.py`: script to generate mock CSV data
- `src/train_logistic_regression.py`: script to train and evaluate model

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate (or overwrite) mock CSV data:

```bash
python src/generate_mock_data.py
```

Train and evaluate logistic regression model:

```bash
python src/train_logistic_regression.py
```
