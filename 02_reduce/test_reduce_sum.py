#
import torch
import torch_reduce_sum


def test():
    x = torch.ones((10000000,), device="cuda")
    print(x)
    # 自定义实现
    import time

    ## warm up
    for i in range(10):
        torch_reduce_sum.reduce_sum(x, 0)
        torch_reduce_sum.reduce_sum(x, 1)
    torch.cuda.synchronize()

    start = time.time()
    for i in range(10):
        y_custom0 = torch_reduce_sum.reduce_sum(x, 0)
        torch.cuda.synchronize()
    print(f"自定义0 时间: {time.time() - start:.4f}")
    start = time.time()
    for i in range(10):
        y_custom1 = torch_reduce_sum.reduce_sum(x, 1)
        torch.cuda.synchronize()

    print(f"自定义1 时间: {time.time() - start:.4f}")

    # PyTorch原生实现
    y_ref = x.sum()
    print(f"自定义: {y_custom1.item():.4f}, 自定义: {y_custom0.item():.4f}")
    # 结果验证
    assert torch.allclose(y_custom1, y_ref, rtol=1e-4), "结果不一致"
    print(f"测试通过! 自定义: {y_custom1.item():.4f}, 参考: {y_ref.item():.4f}")


if __name__ == "__main__":
    test()
