import os
from huggingface_hub import HfApi,create_repo
token=os.environ["HF_TOKEN"]; space=os.environ["HF_SPACE_REPO"]
create_repo(space,repo_type="space",space_sdk="docker",exist_ok=True,token=token)
HfApi(token=token).upload_folder(folder_path="deployment",repo_id=space,repo_type="space")
