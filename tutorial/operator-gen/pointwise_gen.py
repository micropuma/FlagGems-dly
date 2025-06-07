import triton
import torch
import flag_gems
from flag_gems.utils import pointwise_dynamic

# =========================== 算子生成sample =====================================

import os
os.environ["FLAGGEMS_DEBUG"] = "1"  # 启用调试模式

@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def abs_func(x):
    return tl.abs(x)

@pointwise_dynamic(
    is_tensor=[True, True, False],
    dtypes=[None, None, float],
    promotion_methods=[(0,"DEFAULT")]
)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha

x = torch.tensor([1, 2, 3], dtype=torch.float32).to(device='cuda')
y = torch.tensor([4, 5, 6], dtype=torch.float32).to(device='cuda')
alpha = 2.0
result = add_func(x, y, alpha)
print(result)  # Should print tensor([ 9., 12., 15.])

# =========================== 性能测试 =====================================
import time

# 测试性能
torch_start = time.time()
torch_result = x + y * alpha
torch_time = time.time() - torch_start

flag_start = time.time()
flag_result = add_func(x, y, alpha)
flag_time = time.time() - flag_start

print(f"PyTorch 耗时: {torch_time*1000:.2f}ms")
print(f"FlagGems 耗时: {flag_time*1000:.2f}ms")
print(f"加速比: {torch_time/flag_time:.1f}x")
print("结果一致:", torch.allclose(torch_result, flag_result))



