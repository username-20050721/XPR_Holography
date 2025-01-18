from wtconv import WTConv2d
import torch


if __name__ == "__main__":
    test_tensor = torch.randn(4, 3, 2160, 3840)
    conv_dw = WTConv2d(4, 4, kernel_size=5, wt_levels=5)
    test_tensor = conv_dw(test_tensor)
    print(test_tensor.size())