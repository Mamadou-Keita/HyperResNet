import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

  def __init__(self, in_chan, out_chan, stride= 1, downsample=None):
    super().__init__()

    self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bachNorm1 = nn.BatchNorm2d(out_chan)
    self.actReLU = nn.ReLU(inplace = True)

    self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride= 1, padding=1, bias=False)
    self.bachNorm2 = nn.BatchNorm2d(out_chan)

    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    skip = x

    out = self.conv1(x)
    out = self.bachNorm1(out)
    out = self.actReLU(out)
    
    out = self.conv2(out)
    out = self.bachNorm2(out)

    if self.downsample is not None:
      skip = self.downsample(x)

    out = out + skip
    out = self.actReLU(out)

    return out


class HyperResNet(nn.Module):

  def __init__(self, BasicBlock, layers, nbClasses):
    super().__init__()

    self.inchannel = 64


    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(self.inchannel)
    self.relu = nn.ReLU(inplace= True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    self.layer1 = self._resnet_layer(BasicBlock, 64, layers[0])
    self.layer2 = self._resnet_layer(BasicBlock, 128, layers[1], stride=2)
    self.layer3 = self._resnet_layer(BasicBlock, 256, layers[2], stride=2)
    self.layer4 = self._resnet_layer(BasicBlock, 512, layers[3], stride=2)

    self.conv2_3 = self.downgrade(64, 128)
    self.conv3_4 = self.downgrade(128, 256)
    self.conv4_5 = self.downgrade(256, 512)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 , nbClasses)


  def downgrade(self, inplanes, outplanes):
  
    return nn.Sequential(
        nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(inplanes),
        nn.ReLU(inplace= True),

        nn.Conv2d(inplanes, outplanes, kernel_size=1),
        nn.BatchNorm2d(outplanes)
      )
  
  def _resnet_layer(self, BasicBlock, outChannel, Blocks, stride=1):
    downsample = None

    if stride != 1 or self.inchannel != outChannel:

      downsample = nn.Sequential(
          nn.Conv2d(self.inchannel, outChannel, kernel_size= 1, stride= stride, bias= False),
          nn.BatchNorm2d(outChannel)
      )

    layers = []
    layers.append(BasicBlock(self.inchannel, outChannel, stride, downsample))
    self.inchannel = outChannel

    for index in range(1, Blocks):

      layers.append(BasicBlock(self.inchannel, outChannel))


    return nn.Sequential(*layers)


  def forward(self, x):

    x = self.conv1(x)           
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)   
    temp =  x
    x = self.layer2(x)
    temp = self.conv2_3(temp) + x 
    x = self.layer3(x)
    temp = self.conv3_4(temp) + x
    x = self.layer4(x) 
    temp = self.conv4_5(temp) + x

    x = self.avgpool(temp)
    x = torch.flatten(x, 1)

    x = self.fc(x)

    return x


def HyperResNet34(nbClass):
    layers=[3, 4, 6, 3]
    model = HyperResNet(ResidualBlock, layers, nbClass)
    return model