import torch

print("PyTorch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
	print("GPU count:", torch.cuda.device_count())
	print("Current device:", torch.cuda.current_device())
	print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
	print("Hint: if Torch CUDA build is None, install GPU wheels from requirements.torch.cu121.txt")