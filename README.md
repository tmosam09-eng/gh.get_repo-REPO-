# Tourism Purchase Prediction — MLOps Project

Target: `ProdTaken`. Raw index/ID fields are removed; `Fe Male` is standardized to `Female`; split is stratified 80/20.

## Validated local Random Forest
Accuracy 0.902 | Precision 0.811 | Recall 0.639 | F1 0.715 | ROC-AUC 0.937

The production scripts use Hugging Face Datasets/Hub, MLflow, GridSearchCV, Docker, Streamlit, and GitHub Actions.
Set GitHub secrets: `HF_TOKEN`, `HF_DATASET_REPO`, `HF_MODEL_REPO`, `HF_SPACE_REPO`.
