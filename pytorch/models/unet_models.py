import torch
from torchvision import transforms


class DoubleConv2D(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), padding="same"),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), padding="same"),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(torch.nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=None, fixed_upsampling=True):
        super(UNet2D, self).__init__()
        if features is None:
            features = [128, 128, 128, 128]
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv2D(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            if not fixed_upsampling:
                self.ups.append(torch.nn.ConvTranspose2d(feature, feature, kernel_size=2, stride=2))
            else:
                self.ups.append(torch.nn.UpsamplingBilinear2d(scale_factor=2))
            self.ups.append(DoubleConv2D(feature * 2, feature))

        self.bottleneck = DoubleConv2D(features[-1], features[-1])
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1, padding="same", bias=False)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        channel_dimension = len(x.shape) - 3  # if we have batches (4 dim tensor) this is 1, if not (3dim) 0

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=channel_dimension)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
