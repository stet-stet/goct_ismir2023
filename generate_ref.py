"""
python generate_ref.py +ts_dataset=beatfine
"""
import torch 
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import time
import hydra
import numpy as np 
from omegaconf import OmegaConf
import os
import logging
from pprint import pprint
from code.counter.osu4kcounter import Osu4kTwoBeatOnsetCounter

logger = logging.getLogger(__file__)

trunc_how_many = None
global_counter = Osu4kTwoBeatOnsetCounter()

def generate_2bar_greedy(tokens):
    """
    ! invariants
    tokens: (1, length_in_beats//2 - 1, 100). includes charts from beat 2. beats [2-4, 4-6, 6-8, ....]
    """
    global trunc_how_many

    generated = []
    how_many_twobeats = tokens.shape[1]

    for idx_to_generate_rn in range(0,how_many_twobeats):
        logger.info(f"{idx_to_generate_rn}\t\t{tokens[:,idx_to_generate_rn,:]}")
    
    logger.info(f"precision {None} \trecall {None} \tf1 {None}")
    return generated
    


def run(args):
    from code import DatasetSelector 
    global trunc_how_many
    global global_counter
    ts_loader = DataLoader(
        DatasetSelector(args.ts_dataset)(),
        batch_size=1, 
        shuffle=False
    )
    trunc_how_many = 92

    ts_pbar = tqdm(ts_loader)
    
    for data, tokens, fine_difficulty,songname in ts_pbar:
        data = data[:].to('cuda:0').to(torch.long) #(1, 80, 96T)
        tokens= tokens.to(torch.long).to('cuda:0') # (1, tokenlength)
        fine_difficulty = fine_difficulty.to('cuda:0').to(torch.float).unsqueeze(1) # (1, 1)

        logger.info(songname)
        generated = generate_2bar_greedy(tokens)
        continue
                
    print("====subtotl====")
    logger.info(f"precision {None} \trecall {None} \tf1 {None}")



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
        

