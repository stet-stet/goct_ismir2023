from omegaconf import OmegaConf
from .audiochartfinebeatdataset import AudioChartFineBeatDataset
from .audiotimingfinebeatdataset import AudioTimingFineBeatDataset
from .audiorandomshift import AudioRandomShiftDataset
from .audiocharttwobeatsongdataset import AudioChartTwoBeatSongDataset
from .audiotimingtwobeatsongdataset import AudioTimingTwoBeatSongDataset
from .ddcdataset import DDCDataset

from ..utils.creator_factory import makeClassMaker

def DatasetSelector(args):
    """
    :params: args: actually either args.tr_loader, args.cv_loader args.ts_loaser
    """
    print("dataset\t:",args)
    b = OmegaConf.to_container(args)
    if args.name == "AudioChartFineBeatDataset": # **mel
        return makeClassMaker(AudioChartFineBeatDataset, **b['parameters'])
    elif args.name == "AudioTimingFineBeatDataset":
        return makeClassMaker(AudioTimingFineBeatDataset, **b['parameters']) # timingonly
    elif args.name == "AudioRandomShiftDataset": # mel, random shift
        return makeClassMaker(AudioRandomShiftDataset, **b['parameters']) 
    elif args.name == "AudioChartTwoBeatSongDataset": # **mel, eval
        return makeClassMaker(AudioChartTwoBeatSongDataset, **b['parameters'])
    elif args.name == "AudioTimingTwoBeatSongDataset": # fraxtil, itg timingonly, eval
        return makeClassMaker(AudioTimingTwoBeatSongDataset, **b['parameters'])
    elif args.name == "DDCDataset":
        return makeClassMaker(DDCDataset, **b['parameters'])
