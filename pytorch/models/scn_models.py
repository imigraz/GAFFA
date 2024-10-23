from pytorch.models.unet_models import *


class Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                 activation=torch.nn.LeakyReLU(0.1, inplace=True)):
        super(Conv2D, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding="same"),
            torch.nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        return self.conv(x)


class SCN2D(torch.nn.Module):

    def __init__(self, in_channels=1, out_channels=1, local_component_features=None, spatial_component_features=None
                 , fixed_upsampling=True):
        super(SCN2D, self).__init__()
        if local_component_features is None:
            local_component_features = [128, 128, 128, 128]
        if spatial_component_features is None:
            spatial_component_features = [128, 128, 128]

        self.LocalComponent = UNet2D(in_channels, out_channels, local_component_features)
        self.pool2d = torch.nn.AvgPool2d(kernel_size=3, stride=4)

        self.SpatialComponent = torch.nn.ModuleList()
        spatial_conv_kernel_size = [7, 7]

        in_channels = out_channels
        for feature in spatial_component_features:
            self.SpatialComponent.append(Conv2D(in_channels=in_channels, out_channels=feature,
                                                kernel_size=spatial_conv_kernel_size))
            in_channels = feature
        self.SpatialComponent.append(Conv2D(in_channels=spatial_component_features[-1], out_channels=out_channels,
                                            kernel_size=spatial_conv_kernel_size, activation=torch.nn.Tanh()))

        if not fixed_upsampling:
            self.up2d = torch.nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4,
                                                 stride=4)
        else:
            self.up2d = torch.nn.Upsample(mode="bicubic", scale_factor=4)

    def forward(self, x):
        local_component = self.LocalComponent(x)

        spatial_component = self.pool2d(local_component)
        for layer in self.SpatialComponent:
            spatial_component = layer(spatial_component)
        spatial_component = self.up2d(spatial_component)

        return local_component * spatial_component, local_component, spatial_component
