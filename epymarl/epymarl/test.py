import torch
print(torch.__version__)  # 查看 PyTorch 版本
print(torch.cuda.is_available())  # 是否支持 CUDA
print(torch.version.cuda)  # 查看 PyTorch 编译时使用的 CUDA 版本
print(torch.cuda.device_count())  # 可用 GPU 数量
print(torch.cuda.get_device_name(0))  # 获取 GPU 名称（如果有多个 GPU，可尝试 1, 2, ...）
print(torch.cuda.current_device())
x = torch.randn(3, 3).cuda()
print(x) 

