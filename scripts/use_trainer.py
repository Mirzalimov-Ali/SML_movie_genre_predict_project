import pandas as pd
import os
from joblib import dump
from rich.console import Console
from rich.table import Table
from collections import Counter

# sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# src
from src.trainer import Trainer
from src.logger import get_logger

logger = get_logger('use_trainer', 'model_training.log')

# ____________________________________________ Load dataset ____________________________________________

df = pd.read_csv('data/preprocessed/preprocessed_dataset.csv')
df.drop('Id', axis=1, inplace=True)
x = df.drop('Genre', axis=1)
y = df['Genre']

logger.info(f"Dataset loaded. Shape: {df.shape}. Classes distribution: {Counter(y)}")

# ____________________________________________ Define models ____________________________________________

models = [
    LogisticRegression(max_iter=1000, random_state=42),
    RandomForestClassifier(n_estimators=200, random_state=42),
    ExtraTreesClassifier(n_estimators=200, random_state=42),
    GradientBoostingClassifier(random_state=42),
    HistGradientBoostingClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    LGBMClassifier(random_state=42),
    CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=0, random_state=42),
    KNeighborsClassifier(n_neighbors=10),
    BaggingClassifier(random_state=42)
]

# ____________________________________________ Train & Evaluate ____________________________________________

results = []
best_acc = -float("inf")
best_model = None
best_model_name = ""

os.makedirs('model/best', exist_ok=True)
os.makedirs('model/others', exist_ok=True)

for model in models:
    trainer = Trainer(model, x, y)
    trainer.train()
    res = trainer.evaluate()

    results.append([
        model.__class__.__name__,
        res["accuracy"],
        res["kfold_mean"],
        res["kfold_std"],
        len(res["selected_features"])
    ])

    # save all trained models
    model_path = f"model/others/{model.__class__.__name__}.joblib"
    dump(trainer.clf, model_path)

    # track best model
    if res["accuracy"] > best_acc:
        best_acc = res["accuracy"]
        best_model = trainer.clf
        best_model_name = model.__class__.__name__

# ____________________________________________ Create comparison table ____________________________________________

console = Console()
results_sorted = sorted(results, key=lambda i: i[1], reverse=True)
best_row = results_sorted[0]
worst_row = results_sorted[-1]

table = Table(title="Model Comparison with SMOTE + OVO + Embedded Feature Selection", show_lines=True)
table.add_column("Algorithm")
table.add_column("Accuracy")
table.add_column("K-Fold Mean")
table.add_column("K-Fold Std")
table.add_column("Selected Features Count")

for row in results_sorted:
    name, acc, kmean, kstd, fcount = row
    if row == best_row:
        table.add_row(f"[bold green]{name}[/bold green]",
                      f"[bold green]{acc:.2f}[/bold green]",
                      f"[bold green]{kmean:.2f}[/bold green]",
                      f"[bold green]{kstd:.2f}[/bold green]",
                      f"[bold green]{fcount}[/bold green]")
    elif row == worst_row:
        table.add_row(f"[bold red]{name}[/bold red]",
                      f"[bold red]{acc:.2f}[/bold red]",
                      f"[bold red]{kmean:.2f}[/bold red]",
                      f"[bold red]{kstd:.2f}[/bold red]",
                      f"[bold red]{fcount}[/bold red]")
    else:
        table.add_row(name, f"{acc:.2f}", f"{kmean:.2f}", f"{kstd:.2f}", str(fcount))

# ____________________________________________ Save table ____________________________________________

temp_console = Console(record=True)
temp_console.print(table)
text = temp_console.export_text()
with open("results/all_model_compare.txt", "w", encoding="utf-8") as f:
    f.write(text)
logger.info("Comparison table saved at results/all_model_compare.txt")

# ____________________________________________ Save best model ____________________________________________

if best_model is not None:
    best_model_path = f"model/best/{best_model_name}.joblib"
    others_path = f"model/others/{best_model_name}.joblib"
    if os.path.exists(others_path):
        os.remove(others_path)
    dump(best_model, best_model_path)
    logger.info(f"Best model '{best_model_name}' saved at '{best_model_path}' with Accuracy={best_acc:.4f}")
else:
    logger.error("No best model selected.")
