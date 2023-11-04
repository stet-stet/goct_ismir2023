"""
This file used to make the lower two rows of Table 2.

python metrics_timing_AR.py +model=finediffdecoderprelnbipasttiming +ts_dataset=beatfine +abridge=False +ckpt_path=ckpts/mel_timingonly/0.0002_and_32/ckpt_epoch_9.pth 
precision 0.8889249703349243      recall 0.8062742555457787       f1 0.8455847640214559

python metrics_timing_AR.py +model=finediffdecoderprelnbipasttiming +ts_dataset=beatfine_itg_timingonly +ckpt_path=ckpts/itg_mel_timingonly_finetune/2e-05_and_32/ckpt_epoch_13.pth
precision 0.7074169077116192      recall 0.783818104855778        f1 0.7436603557085484

python metrics_timing_AR.py +model=finediffdecoderprelnbipasttiming +ts_dataset=beatfine_fraxtil_timingonly +ckpt_path=vckpts/fraxtil_mel_timingonly_finetune/2e-05_and_32/ckpt_epoch_13.pth
precision 0.7460073034449382      recall 0.8383683559950557       f1 0.7894957396284398

python metrics_timing_AR.py +model=finediffdecoderprelnbipasttiming +ts_dataset=beatfine_itg_timingonly +ckpt_path=ckpts/itg_mel_timingonly/0.0002_and_32/ckpt_epoch_9.pth
precision 0.6625449640287769      recall 0.7128257846042209       f1 0.6867662908910833 

python metrics_timing_AR.py +model=finediffdecoderprelnbipasttiming +ts_dataset=beatfine_fraxtil_timingonly +ckpt_path=ckpts/fraxtil_mel_timingonly/0.0002_and_32/ckpt_epoch_9.pth
precision 0.6798731442754056      recall 0.7313720642768851       f1 0.7046829593635368

I am sorry for not setting the seed properly...
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
from code.counter.osu4kcounter import Osu4kTwoBeatOnsetCounter, Osu4kTwoBeatTimingCounter

logger = logging.getLogger(__file__)

trunc_how_many = None
global_counter = Osu4kTwoBeatTimingCounter()


def generate_2bar_greedy(model, data, tokens, fine_difficulty, max_gen_length=200):
    """
    ! invariants
    model, data, tokens are already in cuda device
    data: (1, 80, 96T)
    tokens: (1, length_in_beats//2 - 1, 100). includes charts from beat 2. beats [2-4, 4-6, 6-8, ....]
    fine_diff: (1,1) 
    data has dim of (80, L)
    """
    global trunc_how_many
    global global_counter


    local_counter = Osu4kTwoBeatTimingCounter()


    decoder_macro_input = [177 for _ in range(max_gen_length//2-1)] + [96,96] + [177 for _ in range(max_gen_length//2-1)]
    decoder_macro_input = decoder_macro_input[trunc_how_many:]
    decoder_macro_input = torch.Tensor(decoder_macro_input).to(torch.long).to('cuda:0').unsqueeze(0)
    generated = []
    how_many_twobeats = tokens.shape[1]

    for idx_to_generate_rn in range(0,how_many_twobeats):
        retrieve_this = (max_gen_length // 2) - trunc_how_many
        where_is_96 = int(retrieve_this)
        audio_this_cycle = data[: , :, 96*(idx_to_generate_rn):96*(idx_to_generate_rn+2)]
        tokens_this_cycle = decoder_macro_input[:,:]
        if audio_this_cycle.shape[2] == 0: break
        while retrieve_this < 199 - trunc_how_many: # nothing's gonna reach 100 anyways, so cutting it short should be fine...right?
            model_output = model(audio_this_cycle, tokens_this_cycle, fine_difficulty)
            created_tokens = torch.argmax(model_output, 2).to(torch.long)
            if int(created_tokens[0,retrieve_this]) == 177:
                break # input_this_cycle is the answer for this cycle.
            else:
                tokens_this_cycle[0,retrieve_this+1] = created_tokens[0,retrieve_this]
                retrieve_this += 1
        # split tokens by 96
        created_tokens = created_tokens[:, where_is_96:retrieve_this]
        local_counter.update(tokens[0,idx_to_generate_rn,:],created_tokens[0,:])
        global_counter.update(tokens[0,idx_to_generate_rn,:],created_tokens[0,:])
        generated.append(created_tokens)
        logger.info(f"{idx_to_generate_rn}\t\t{created_tokens}")
        created_tokens_length = created_tokens.shape[1]

        decoder_macro_input = torch.cat((
                            torch.Tensor([[177 for _ in range(max_gen_length//2-1-created_tokens_length)]]).to(torch.long).to('cuda'),
                            torch.Tensor([[96]]).to(torch.long).to('cuda'),
                            created_tokens,
                            torch.Tensor([[96]]).to(torch.long).to('cuda'),
                            torch.Tensor([[177 for _ in range(max_gen_length//2-1)]]).to(torch.long).to('cuda')
                        ),axis=1)
        how_many_pad_on_right = max_gen_length - decoder_macro_input.shape[1]
        decoder_macro_input = F.pad(decoder_macro_input, (0, how_many_pad_on_right), 'constant', 177)
        decoder_macro_input = decoder_macro_input[:,trunc_how_many:]
    
    logger.info(f"precision {local_counter.precision()} \trecall {local_counter.recall()} \tf1 {local_counter.f1()}")
    assert len(generated) == how_many_twobeats or len(generated) == how_many_twobeats-1
    return generated
    


def run(args):
    from code import DatasetSelector, ModelSelector
    global trunc_how_many
    global global_counter
    ts_loader = DataLoader(
        DatasetSelector(args.ts_dataset)(),
        batch_size=1, 
        shuffle=False
    )
    model = ModelSelector(args.model)().to('cuda:0')
    model.load_state_dict(torch.load(args.ckpt_path))

    model.eval()

    trunc_how_many = 46

    ts_pbar = tqdm(ts_loader)

    abridge = False 
    try:
        if args.abridge is True:
            abridge = True
    except Exception: #omegaconf.errors.ConfigAttributeError:
        abridge = False
    
    for n, (data, tokens, fine_difficulty,songname) in enumerate(ts_pbar):
        data = data[:].to('cuda:0') #(1, 80, 96T)
        tokens= tokens.to(torch.long).to('cuda:0') # (1, tokenlength)
        fine_difficulty = fine_difficulty.to('cuda:0').to(torch.float).unsqueeze(1) # (1, 1)

        logger.info(songname)
        generated = generate_2bar_greedy(model, data, tokens, fine_difficulty, 100)
        if abridge and n > 20:
            break
                
    print("====subtotl====")
    logger.info(f"precision {global_counter.precision()} \trecall {global_counter.recall()} \tf1 {global_counter.f1()}")


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
        

