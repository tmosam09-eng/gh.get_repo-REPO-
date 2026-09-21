import os
from huggingface_hub import HfApi, create_repo
repo=os.environ["HF_DATASET_REPO"]; token=os.environ["HF_TOKEN"]
create_repo(repo,repo_type="dataset",exist_ok=True,token=token)
HfApi(token=token).upload_file(path_or_fileobj="tourism.csv",path_in_repo="raw/tourism.csv",repo_id=repo,repo_type="dataset")
