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
  

class FashionMNISTLinearModel(torch.nn.Module):
  def __init__(self, n_features: int, n_targets: int):
    super(FashionMNISTLinearModel, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(n_features, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, n_targets),
    )

  def forward(self, inputs):
    return self.model.forward(inputs)

  def n_paramaters(self):
    return (
      sum(params.numel() for params in self.model.parameters() if params.requires_grad)
    )
  
class FashionMNISTConvModel(torch.nn.Module):
  def __init__(self, width: int, height: int, n_targets: int):
    super(FashionMNISTConvModel, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(1, n_targets, kernel_size=3, stride=1),
      torch.nn.BatchNorm2d(n_targets),
      torch.nn.ReLU(),
      torch.nn.AvgPool2d(kernel_size=2),
      torch.nn.Conv2d(n_targets, n_targets, kernel_size=2, stride=1),
      torch.nn.BatchNorm2d(n_targets),
      torch.nn.ReLU(),
      torch.nn.AvgPool2d(kernel_size=12),
    )
    
  def forward(self, inputs: torch.Tensor):
    output = torch.squeeze(self.model(inputs))
    return output


  def n_paramaters(self):
    return (
      sum(params.numel() for params in self.model.parameters() if params.requires_grad)
    )


class NextTokenPredictor(torch.nn.Module):
  def __init__(self, n_embeddings, hidden_size=64):
    super(NextTokenPredictor, self).__init__()

    self.gru = torch.nn.Sequential(
      torch.nn.Embedding(n_embeddings, hidden_size),
      torch.nn.GRU(hidden_size, hidden_size // 2, num_layers=1),
    )
    self.generator = torch.nn.Sequential(
      torch.nn.GELU(),
      torch.nn.Linear(hidden_size // 2, n_embeddings),
      torch.nn.Softmax(dim=-1)
    )

  def forward(self, input_ids: torch.Tensor):
    output = self.generator(self.gru(input_ids[0])[1])
    for input in input_ids:
      _, hidden = self.gru(input)
      result = self.generator(hidden)
      output = torch.vstack((output, result))

    return output

class FedAvgCNN(torch.nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(dim, 512),
            torch.nn.ReLU(inplace=True)
        )
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# based on https://arxiv.org/pdf/1704.04861.pdf
class MobilNet(torch.nn.Module):
  block_configs = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

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