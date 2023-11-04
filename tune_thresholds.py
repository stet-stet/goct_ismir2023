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
            

def tune_thres(outputs, targets):
    """
    targets: (B*1000, 112)
    outputs: list, len(outputs) = B*1000, each list in outputs may have variable length.
    """
    B = len(outputs)
    print(f"{B} samples to tune with")
    max_f1 = 0 
    max_thres = 0
    for thres in tqdm([0.01 * i for i in range(30)]):
        fn, fp, tp = 0, 0, 0
        oo = [[e[0] for e in ll if e[1]>thres] for ll in outputs]
        for idx in range(B):
            target_in_question = targets[idx]
            output_in_question = oo[idx]
            fnt, fpt, tpt = get_fn_fp_tp(output_in_question, target_in_question)
            fn += fnt 
            fp += fpt
            tp += tpt
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision+recall + 1e-9)
        if f1 > max_f1:
            max_f1 = f1
            max_thres = thres
        # print("on ", thres, ":", max_thres, " and ", max_f1)
    return max_thres, max_f1
# cnn, /mnt/c/Users/manym/Desktop/gorst/gorst/ckpts/ddc_cnn/2e4_and_64_try2/ckpt_epoch_1.pth
# easy 0.19, 0.745
# normal 0.17, 0.822
# hard 0.17, 0.837
# insane 0.17, 0.840
# expert 0.17, 0.812
# expert+ 0.13, 0.813
    
# clstm, /mnt/c/Users/manym/Desktop/gorst/gorst/ckpts/ddc_clstm/0.0002_and_64/ckpt_epoch_0.pth
# easy 0.17, 0.766
# normal 0.16 0.831
# hard 0.18 0.839
# insane 0.19 0.844
# expert 0.17, 0.822
# expert+ 0.13, 0.814

def run(args):
    from code import DatasetSelector, ModelSelector, LossSelector, OptimizerSelector
    from tqdm import tqdm

    #torch.manual_seed(10101010)
    cv_loader = DataLoader(
        DatasetSelector(args.cv_dataset)(),
        batch_size=args.experiment.batch_size.cv, 
        num_workers=args.experiment.num_workers,
        shuffle=True
    )
    model = ModelSelector(args.model)().to('cuda:0')
    model.load_state_dict(torch.load(args.ckpt_path))

    normalizer = SpectrogramNormalizer().to('cuda:0')

    valid_step = 0
    valid_pbar = tqdm(cv_loader, position=0,leave=True)
    targets = []
    preds = []
    with torch.no_grad():
        for data, target, fine_difficulty in valid_pbar:
            valid_step += 1
            data = data.to('cuda:0') # (B, 112, 80, 3)
            target = target.to(torch.bool) # (B, 112)
            fine_difficulty = fine_difficulty.to('cuda:0').to(torch.float) # (B, 1)

            # print(data.shape, target.shape, fine_difficulty.shape)

            data_norm = normalizer(data)
            #print(data_norm.shape)

            model_output = model(data_norm, fine_difficulty)
            model_output = model_output.cpu().detach().numpy() # (B, 112)
            target  = target.cpu().detach().numpy()

            targets.extend(convert_targets(target))
            preds.extend(convert_outputs(model_output))

            if valid_step >= 2000: # B * 1000 should be enough
                break

    print(len(targets))
    print(len(preds))
    tuned_thres,f1 = tune_thres(preds, targets)

    print(tuned_thres,f1)


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