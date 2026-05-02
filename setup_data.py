import kagglehub  # ! Token required

# Download latest version
path = kagglehub.dataset_download("mdmofazzalhossain789/weizmann-video-dataset")

print("Path to dataset files:", path)
