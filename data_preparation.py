import os, pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
repo=os.environ["HF_DATASET_REPO"]; token=os.environ["HF_TOKEN"]
df=load_dataset(repo,data_files="raw/tourism.csv",split="train",token=token).to_pandas()
df=df.drop(columns=["Unnamed: 0","CustomerID"],errors="ignore").drop_duplicates()
df["Gender"]=df["Gender"].replace({"Fe Male":"Female"})
train,test=train_test_split(df,test_size=.20,random_state=42,stratify=df["ProdTaken"])
os.makedirs("data",exist_ok=True); train.to_csv("data/train.csv",index=False); test.to_csv("data/test.csv",index=False)
api=HfApi(token=token)
for f in ["train.csv","test.csv"]:
 api.upload_file(path_or_fileobj=f"data/{f}",path_in_repo=f"processed/{f}",repo_id=repo,repo_type="dataset")
