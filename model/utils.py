from torchvision import models
from torchvision import transforms as T

BACKBONE = {
    'resnet18': {
        'model': models.resnet18,
        'in_features': 512
    },
    'resnet34': {
        'model': models.resnet34,
        'in_features': 512
    },
    'resnet50': {
        'model': models.resnet50,
        'in_features': 2048
    },
    'resnet101': {
        'model': models.resnet101,
        'in_features': 2048
    }
}

preprocessing = T.Compose([
                    T.ToTensor(),
                    T.Resize((112, 112)),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])