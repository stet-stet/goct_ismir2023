import os
import logging
import hydra
import time
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from omegaconf import OmegaConf 

from code.model.cnn import SpectrogramNormalizer, PlacementCNN

logger = logging.getLogger(__name__)

class Tracker():
    def __init__(self) -> None:
        self.initialize()

    def initialize(self):
        self.count = 0
        self.sum = 0.

    def __str__(self):
        if self.count == 0:
            return str(0.)
        else:
            return str(float(self.sum / self.count))
        
    def insert(self, num):
        self.sum += float(num)
        self.count += 1

    def value(self):
        return float(self.sum / self.count)

class RunningMean():
    def __init__(self, howmany=1000) -> None:
        self.howmany=howmany
        self.count = 0
        self.arr = [1. for _ in range(self.howmany)]

    def __str__(self):
        return str(float(sum(self.arr) / len(self.arr)))
    
    def insert(self, num):
        self.arr[self.count % self.howmany] = float(num)
        self.count += 1

    def value(self):
        return float(sum(self.arr) / len(self.arr))

def run(args):
    from code import DatasetSelector, ModelSelector, LossSelector, OptimizerSelector
    from tqdm import tqdm

    #torch.manual_seed(10101010)
    wandb.login()
    run = wandb.init(
    # Set the project where this run will be logged
        project="osu-mania-ddc",
    # Track hyperparameters and run metadata
        config={
            "learning_rate": 1e-3,
            "epochs": 5,
            "testrun": 0,
        }
    )

    tr_loader = DataLoader(
        DatasetSelector(args.tr_dataset)(),
        batch_size=args.experiment.batch_size.tr, 
        num_workers=args.experiment.num_workers,
        shuffle=True
    )
    cv_loader = DataLoader(
        DatasetSelector(args.cv_dataset)(),
        batch_size=args.experiment.batch_size.cv, 
        num_workers=args.experiment.num_workers,
        shuffle=True
    )
    model = ModelSelector(args.model)().to('cuda:0')
    loss = LossSelector(args.loss)().to('cuda:0')
    optimizer = OptimizerSelector(args.optimizer)(model.parameters())

    normalizer = SpectrogramNormalizer().to('cuda:0')

    ckpt_path = str(args.ckpt_path)
    ckpt_path = os.path.join(ckpt_path, f"{str(args.optimizer.parameters.lr)}_and_{str(args.experiment.batch_size.tr)}")
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), f"{ckpt_path}/asdf.pth")
    for epoch in range(20):
        st = time.time()
        tr_pbar = tqdm(tr_loader, position=0,leave=True)
        tracker = Tracker()
        train_step, valid_step = 0, 0
        tr_running_mean = RunningMean(500)
        cv_running_mean = RunningMean(300)
        print("=================================epoch ",epoch)
        for data, target, fine_difficulty in tr_pbar:
            train_step += 1
            data = data.to('cuda:0') # (B, 112, 80, 3)
            target = target.to(torch.float).to('cuda:0') # (B, 112)
            fine_difficulty = fine_difficulty.to('cuda:0').to(torch.float) # (B)

            data_norm = normalizer(data)

            model_output = model(data_norm, fine_difficulty)

            
            
            l = loss(model_output, target)
            l.backward()
            loss_value = float(l)
            tracker.insert(l)
            tr_running_mean.insert(l)
            optimizer.step()
            optimizer.zero_grad()
            if tr_running_mean.count % 160 == 159:
                tr_pbar.set_description(str(loss_value))
                wandb.log({
                    "training_loss": tr_running_mean.value()
                })
            if train_step >= 20000: # 5000 steps per epoch
                break
        print("av.training loss for epoch: ", tracker)
        print(f"train time: {time.time()-st}")
        valid_pbar = tqdm(cv_loader, position=0,leave=True)
        tracker = Tracker()
        with torch.no_grad():
            for data, target, fine_difficulty in valid_pbar:
                valid_step += 1
                data = data.to('cuda:0') # (B, 112, 80, 3)
                target = target.to(torch.float).to('cuda:0') # (B, 112)
                fine_difficulty = fine_difficulty.to('cuda:0').to(torch.float) # (B, 1)


                data_norm = normalizer(data)

                model_output = model(data_norm, fine_difficulty)
                
                l = loss(model_output, target)
                loss_value = float(l)
                tr_pbar.set_description(str(loss_value))
                tracker.insert(l)
                cv_running_mean.insert(l)
                if tracker.count % 300 == 0:
                    print(f"========================== {tracker.count // 300}")
                    print("GT")
                    print(target)
                    print("PRED")
                    print(model_output)
                if valid_step > 1600: break
            end = time.time()
            print(f"epoch time: {end-st}")
            print("av.valid loss for epoch: ", tracker)
            wandb.log({
                "validation_loss": tracker.value()
            })
        print("end epoch. saving....")
        torch.save(model.state_dict(), f"{ckpt_path}/ckpt_epoch_{epoch}.pth")
        print("...saved.")
    print(f"workers: {args.experiment.num_workers}")


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