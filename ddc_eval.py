"""
This file used to make the upper two rows of Table 1.

python ddc_eval.py +ts_dataset=ddc +model=ddc_cnn +experiment=ddc +ckpt_path=ckpts/ddc_cnn/0.0002_and_64/ckpt_epoch_1.pth
precision 0.8404987330632542            , recall 0.8206751211129037             , f1 0.8304686440887686

python ddc_eval.py +ts_dataset=ddc +model=ddc_clstm +experiment=ddc +ckpt_path=ckpts/ddc_clstm/0.0002_and_64/ckpt_epoch_0.pth
precision 0.8390783855934946            , recall 0.8380883363992774             , f1 0.8385830682781487

When you run the eval, you will want to switch out the thresholds as indicated on the comments for function `threshold_from_finediff`.
These thresholds were found using scripts/thres_tune_{clstm,cnn}.sh  
"""
import os
import logging
import hydra
import time
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from omegaconf import OmegaConf 
import copy

from code.model.cnn import SpectrogramNormalizer, PlacementCNN

logger = logging.getLogger(__name__)

def convert_outputs(outputs):
    """
    outputs: (B, 112)
    """
    B = outputs.shape[0]
    ret = [[] for _ in range(B)]
    for i in range(B):
        for j in range(112):
            if j==0 and outputs[i][j] > outputs[i][j+1]:
                ret[i].append((j,outputs[i][j]))
            elif j==111 and outputs[i][j] > outputs[i][j-1]:
                ret[i].append((j,outputs[i][j]))
            elif 0<j and j<111 and outputs[i][j] > outputs[i][j+1] and outputs[i][j] > outputs[i][j-1]:
                ret[i].append((j,outputs[i][j]))
    return ret

def convert_targets(targets):
    return list(targets)

def threshold_from_finediff(finediffs):
    # cnn. /mnt/c/Users/manym/Desktop/gorst/gorst/ckpts/ddc_cnn/2e4_and_64_try2/ckpt_epoch_1.pth
    # easy 0.19, 0.745
    # normal 0.17, 0.822
    # hard 0.17, 0.837
    # insane 0.17, 0.840
    # expert 0.17, 0.812
    # expert+ 0.13, 0.813

    # clstm. /mnt/c/Users/manym/Desktop/gorst/gorst/ckpts/ddc_clstm/0.0002_and_64/ckpt_epoch_0.pth
    # easy 0.17, 0.766
    # normal 0.16 0.831
    # hard 0.18 0.839
    # insane 0.19 0.844
    # expert 0.17, 0.822
    # expert+ 0.13, 0.814
    #print(finediffs)
    ret = [0.17 for e in finediffs]
    for n,finediff in enumerate(finediffs):
        if finediff < 2.0: ret[n] = 0.17
        elif 2.0<=finediff and finediff < 2.7: ret[n] = 0.16
        elif 2.7<=finediff and finediff < 4.0: ret[n] = 0.18
        elif 4.0<=finediff and finediff < 5.3: ret[n] = 0.19
        elif 5.3<=finediff and finediff < 6.5: ret[n] = 0.17
        elif finediff >= 6.5: ret[n] = 0.13
    return ret

def get_fn_fp_tp(one_output, one_target):
    """
    output: list of ints from [0,112)
    target: true-false array
    """
    fn, fp, tp = 0, 0, 0
    #print(one_output, one_target)
    for i in range(112):
        if bool(one_target[i]) is False: continue
        for j in [i-2,i-1,i,i+1,i+2]:
            if j in one_output: 
                #print(f"broken on {i}")
                break
            if j == i+2: 
                #print("fn")
                fn += 1

    for peak in one_output:
        flag=False
        for j in [peak-2, peak-1, peak, peak+1, peak+2]:
            if j < 0: continue
            elif j >= 112: 
                #print(f"broken on {peak}")
                break
            if bool(one_target[j]) is True:
                #print("tp")
                tp += 1
                flag=True
                break
        if flag is False: 
            #print("fp")
            fp += 1
    #print(fn, fp, tp )
    return fn, fp, tp
            

def get_metric(outputs, targets, thresholds):
    """
    targets: (B*1000, 112)
    outputs: list, len(outputs) = B*1000, each list in outputs may have variable length.
    """
    B = len(outputs)
    fn, fp, tp = 0, 0, 0
    for output, target, threshold in zip(outputs, targets, thresholds):
        oo = [e[0] for e in output if e[1]>threshold]
        fnt,fpt,tpt = get_fn_fp_tp(oo, target)
        fn+=fnt
        fp+=fpt
        tp+=tpt
    return fn,fp,tp


def run(args):
    from code import DatasetSelector, ModelSelector, LossSelector, OptimizerSelector
    from tqdm import tqdm

    #torch.manual_seed(10101010)
    ts_loader = DataLoader(
        DatasetSelector(args.ts_dataset)(),
        batch_size=args.experiment.batch_size.ts, 
        num_workers=args.experiment.num_workers,
        shuffle=True
    )
    model = ModelSelector(args.model)().to('cuda:0')
    model.load_state_dict(torch.load(args.ckpt_path))

    normalizer = SpectrogramNormalizer().to('cuda:0')
    fn, fp, tp = 0, 0, 0
    the_step = 0
    the_pbar = tqdm(ts_loader, position=0,leave=True)

    with torch.no_grad():
        for data, target, fine_difficulty in the_pbar:
            the_step += 1
            data = data.to('cuda:0') # (B, 112, 80, 3)
            target = target.to(torch.bool) # (B, 112)
            fine_difficulty = fine_difficulty.to('cuda:0').to(torch.float) # (B, 1)

            # print(data.shape, target.shape, fine_difficulty.shape)

            data_norm = normalizer(data)
            #print(data_norm.shape)

            model_output = model(data_norm, fine_difficulty)

            model_output = model_output.cpu().detach().numpy() # (B, 112)
            target  = target.cpu().detach().numpy()
            fine_difficulty = fine_difficulty.squeeze().cpu().detach().numpy()

            target = convert_targets(target)
            pred = convert_outputs(model_output)
            thresholds = threshold_from_finediff(fine_difficulty)

            fnt,fpt,tpt = get_metric(pred,target,thresholds)
            fn += fnt
            fp += fpt
            tp += tpt

            # if valid_step >= 2000: 
            #     break

        print(fn, fp, tp)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision+recall + 1e-9)
    print(f"precision {precision}\t\t, recall {recall}\t\t, f1 {f1}")


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.info(args)
    run(args)

print(__file__)
this_script_dir = os.path.dirname(__file__)

@hydra.main(version_base=None, config_path=os.path.join(this_script_dir,'conf'))
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("some error happened")
        os._exit(1)

if __name__=="__main__":
    main()