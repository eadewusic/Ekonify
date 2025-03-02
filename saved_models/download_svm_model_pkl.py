# My svm model is 10.0 GB which exceeds GitHub storage and LFS
# Hence using Google Drive approach

import gdown

file_id = "1O9uT3LNU2F2QAp6WRmVV2zo1kV37suA-"
output = "saved_models/svm_model.zip"

print("Downloading model...")
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
print("Download complete!")
