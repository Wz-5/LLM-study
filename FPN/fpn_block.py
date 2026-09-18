from torch import nn

class TopdownBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init_()
        self.lateral_comv=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        lateral=self.lateral_comv(x)
        top_down=nn.functional.interpolate(lateral,size=x.shape[-2:],mode='nearest')
        return top_down +lateral