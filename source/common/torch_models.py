import torch

class MNISTModel(torch.nn.Module):
  def __init__(self, n_features: int, n_targets: int):
    super(MNISTModel, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(n_features, 32),
      torch.nn.Sigmoid(),
      torch.nn.Linear(32, n_targets),
    )

  def forward(self, inputs):
    return self.model.forward(inputs)
  

class SimpleLinear(torch.nn.Module):
  def __init__(self, n_features: int, n_targets: int):
    super(SimpleLinear, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(n_features, n_targets)
    )

  def forward(self, inputs):
    return self.model.forward(inputs)
  

class FashionMNISTModel(torch.nn.Module):
  def __init__(self, n_features: int, n_targets: int):
    super(FashionMNISTModel, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(n_features, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, n_targets),
    )

  def forward(self, inputs):
    return self.model.forward(inputs)


# based on https://arxiv.org/pdf/1704.04861.pdf
class MobilNet(torch.nn.Module):
  # block_configs = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
  block_configs = [(64, 2), (128,4)]


  def __init__(self, num_classes=10):
    super(MobilNet, self).__init__()

    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
      torch.nn.BatchNorm2d(32),
      torch.nn.ReLU(),
      *self._blocks(in_channels=32),
      torch.nn.AvgPool2d(kernel_size=4)
    )
    self.linear = torch.nn.Linear(128, num_classes)

  def _blocks(self, in_channels):
    layers = []
    for x in self.block_configs:
      out_channels = x if isinstance(x, int) else x[0]
      stride = 1 if isinstance(x, int) else x[1]
      layers.append(MobilNet.Block(in_channels, out_channels, stride))
      in_channels = out_channels
    return torch.nn.Sequential(*layers)

  def forward(self, inputs):
    out: torch.Tensor = self.model(inputs)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out

  def n_paramaters(self):
    return (
      sum(params.numel() for params in self.model.parameters() if params.requires_grad) +
      sum(params.numel() for params in self.linear.parameters() if params.requires_grad)
    )

  class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
      super(MobilNet.Block, self).__init__()

      self.model = torch.nn.Sequential(
        torch.nn.Conv2d(
          in_channels,
          in_channels,
          kernel_size=3,
          stride=stride,
          padding=1,
          groups=in_channels,
          bias=False
        ),
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(
          in_channels,
          out_channels,
          kernel_size=1,
          stride=1,
          padding=0,
          bias=False
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
      )

    def forward(self, inputs):
      return self.model(inputs)