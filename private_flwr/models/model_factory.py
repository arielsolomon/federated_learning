import torch.nn
from models.resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110
from models.sixty_min_blitz_cnn import Net

models_hub: dict[str, torch.nn.Module] = {
    'blitz': Net(),
    'resnet20': resnet20(),
    'resnet32': resnet32(),
    'resnet44': resnet44(),
    'resnet56': resnet56(),
    'resnet110': resnet110()
}


def get_model_hub_names() -> list[str]:
    return list(models_hub.keys())


def get_model(model_name: str = 'blitz', device: torch.device = 'cpu') -> torch.nn.Module:
    assert model_name in models_hub, f'Expected one of {models_hub.keys()}.\n Got {model_name}'
    model = models_hub[model_name]
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
    return model.to(device)
