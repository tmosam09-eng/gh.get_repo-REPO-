import os,json,joblib,mlflow,mlflow.sklearn
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from huggingface_hub import HfApi,create_repo
token=os.environ["HF_TOKEN"]; dr=os.environ["HF_DATASET_REPO"]; mr=os.environ["HF_MODEL_REPO"]
tr=load_dataset(dr,data_files="processed/train.csv",split="train",token=token).to_pandas()
te=load_dataset(dr,data_files="processed/test.csv",split="train",token=token).to_pandas()
Xtr,ytr=tr.drop(columns="ProdTaken"),tr["ProdTaken"]; Xte,yte=te.drop(columns="ProdTaken"),te["ProdTaken"]
cats=Xtr.select_dtypes("object").columns.tolist(); nums=Xtr.select_dtypes(exclude="object").columns.tolist()
pre=ColumnTransformer([("num",SimpleImputer(strategy="median"),nums),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cats)])
pipe=Pipeline([("preprocessor",pre),("model",RandomForestClassifier(random_state=42,class_weight="balanced",n_jobs=-1))])
params={"model__n_estimators":[200,400],"model__max_depth":[None,12],"model__min_samples_split":[2,5],"model__min_samples_leaf":[1,2]}
mlflow.set_experiment("tourism-purchase-prediction")
with mlflow.start_run():
 grid=GridSearchCV(pipe,params,cv=3,scoring="f1",n_jobs=-1); grid.fit(Xtr,ytr); model=grid.best_estimator_
 pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
 m={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"roc_auc":roc_auc_score(yte,prob),"best_cv_f1":grid.best_score_}
 mlflow.log_params(grid.best_params_); mlflow.log_metrics(m); mlflow.sklearn.log_model(model,"model")
 os.makedirs("model_building",exist_ok=True); joblib.dump(model,"model_building/best_model.joblib"); json.dump({"best_params":grid.best_params_,**m},open("model_building/metrics.json","w"),indent=2)
create_repo(mr,repo_type="model",exist_ok=True,token=token); api=HfApi(token=token)
api.upload_file(path_or_fileobj="model_building/best_model.joblib",path_in_repo="best_model.joblib",repo_id=mr,repo_type="model")
api.upload_file(path_or_fileobj="model_building/metrics.json",path_in_repo="metrics.json",repo_id=mr,repo_type="model")
