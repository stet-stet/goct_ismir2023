from omegaconf import OmegaConf
from .gorstconditionininput import GorstFineDiffInDecoderInput
from .gorstconditionininputtiming import GorstFineDiffInDecoderInputTiming
from .cnn import PlacementCNN
from .clstm import PlacementCLSTM

from ..utils.creator_factory import makeClassMaker

def ModelSelector(args):
    """
    :params: args: actually either args.tr_loader, args.cv_loader args.ts_loaser
    """
    print("model\t:",args)
    b = OmegaConf.to_container(args)
    if args.name == "GorstFineDiffInDecoderInputTiming":
        return makeClassMaker(GorstFineDiffInDecoderInputTiming, **b['parameters'])

    if args.name == "GorstFineDiffInDecoderInput": # keep
        return makeClassMaker(GorstFineDiffInDecoderInput, **b['parameters'])
    
    elif args.name == "DDCCNN": #keep
        return makeClassMaker(PlacementCNN, **b['parameters'])

    elif args.name == "DDCCLSTM": #keep
        return makeClassMaker(PlacementCLSTM, **b['parameters'])
    
    raise NotImplementedError("Invalid Model")


