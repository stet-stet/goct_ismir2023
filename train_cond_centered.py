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

    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="osu-mania-4k",
        # Track hyperparameters and run metadata (if you want to)
        config={
            "learning_rate": 2e-4,
            "epochs": 10,
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
    
    ckpt_path = str(args.ckpt_path)
    ckpt_path = os.path.join(ckpt_path, f"{str(args.optimizer.parameters.lr)}_and_{str(args.experiment.batch_size.tr)}")
    if "load_from" in args:
        load_from = str(args.load_from)
        model.load_state_dict(torch.load(load_from))

    os.makedirs(ckpt_path, exist_ok=True)
    for epoch in range(10):
        st = time.time()
        tr_pbar = tqdm(tr_loader, position=0,leave=True)
        tracker = Tracker()
        tr_running_mean = RunningMean(500)
        cv_running_mean = RunningMean(300)
        print("=================================epoch ",epoch)
        for data, tokens, mask, fine_difficulty in tr_pbar:
            data = data.to('cuda:0')
            tokens= tokens.to(torch.long).to('cuda:0')
            fine_difficulty = fine_difficulty.to('cuda:0').to(torch.float).unsqueeze(1)
            mask = mask.unsqueeze(1).to('cuda:0') #(B,1,else)

            model_output = model(data, tokens, fine_difficulty)
            #model_output = model(data,tokens)
            goal = torch.cat((tokens[:,1:], tokens[:,-1:]), axis=1)
            goal_onehot = F.one_hot(goal,num_classes=178).to(torch.float32) # (B, else, C)
            model_output = model_output.permute((0,2,1))[:,:,-50:] #(B,C,else)
            goal_onehot = goal_onehot.permute((0,2,1))[:,:,-50:] #(B,C,else)

            l = loss(model_output,goal_onehot) # (B, C, else)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_value = float(l)
            tr_pbar.set_description(str(loss_value))
            tracker.insert(l)
            tr_running_mean.insert(l)
            if tr_running_mean.count % 500 == 499:
                wandb.log({
                    "training_loss": tr_running_mean.value()
                })
        print("av.training loss for epoch: ", tracker)
        print(f"train time: {time.time()-st}")
        valid_pbar = tqdm(cv_loader, position=0,leave=True)
        tracker = Tracker()
        with torch.no_grad():
            for data, tokens, mask, fine_difficulty in valid_pbar:
                data = data.to('cuda:0')
                tokens= tokens.to(torch.long).to('cuda:0')
                fine_difficulty = fine_difficulty.to('cuda:0').to(torch.float).unsqueeze(1)
                mask = mask.unsqueeze(1).to('cuda:0')

                model_output = model(data, tokens, fine_difficulty) # (32, length, 178)
                # model_output = model(data, tokens)
                goal = torch.cat((tokens[:,1:], tokens[:,-1:]), axis=1)
                goal_onehot = F.one_hot(goal,num_classes=178).to(torch.float32)
                model_output = model_output.permute((0,2,1))[:,:,-50:] #(B,C,else)
                goal_onehot = goal_onehot.permute((0,2,1))[:,:,-50:] #(B,C,else)

                l = loss(model_output,goal_onehot) # (B, C, else)
                loss_value = float(l)
                tr_pbar.set_description(str(loss_value))
                tracker.insert(l)
                cv_running_mean.insert(l)
                if tracker.count % 300 == 0:
                    created_tokens = torch.argmax(model_output.permute((0,2,1)),2).cpu().numpy()
                    print(f"========================== {tracker.count // 300}")
                    print("GT")
                    print(goal[0,:].cpu().numpy())
                    print("PRED")
                    print(created_tokens[0,:])
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