'''
Test script to evaluate GPU access
'''

try:
    import torch
    print('torch version:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print('CUDA device count:', n)
        for i in range(n):
            print(' - device', i, torch.cuda.get_device_name(i))
            # quick GPU allocation test
            a = torch.randn(10, device='cuda:0')
            print('Allocated small tensor on cuda:0, shape:', a.shape)
except Exception as e:
    print('PyTorch check failed or PyTorch not installed:', e)