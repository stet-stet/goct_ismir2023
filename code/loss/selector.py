from omegaconf import OmegaConf
import torch.nn as nn
from ..utils.creator_factory import makeClassMaker

def LossSelector(args):
    """
    :params: args: actually either args.tr_loader, args.cv_loader args.ts_loaser
    """
    print("loss\t:",args)
    b = OmegaConf.to_container(args)
    if args.name == "CrossEntropyLoss":
        return makeClassMaker(nn.CrossEntropyLoss, **b['parameters'])
    if args.name == "BCELoss":
        return makeClassMaker(nn.BCELoss, **b['parameters'])