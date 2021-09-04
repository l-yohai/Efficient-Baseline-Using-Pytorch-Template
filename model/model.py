import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm


class Model(BaseModel):
    def __init__(self, num_classes=10, pretrained_model='efficientnet_b0'):
        self.check_args(pretrained_model)

        super().__init__()
        self.pretrained_model = timm.create_model(
            pretrained_model, pretrained=True)
        # TODO: Your pretrained model's out features
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.pretrained_model(x)
        output = self.fc(output)
        return output

    def check_args(self, pretrained_model):
        if not timm.is_model(pretrained_model):
            assert "Model does not create from timm.\nPlease check the model name."
