import yaml
import torch
import numpy as np
from transformer_lm import TransformerLM
from pathlib import Path
from utils import load_checkpoint, data_loading, data_loader, cross_entropy, save_checkpoint
from adamw import AdamW
from tqdm import tqdm


def load_config(config_path : str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.setup_logging()
        self.setup_dirs()
        
        self.init_model_and_optimizer()
        self.setup_dataloader()
        
        
    def setup_logging(self):
        pass
    
    def setup_dataloader(self):
        train_file_path = self.config['dataset']['train']
        val_file_path = self.config['dataset']['val']
        use_mmap = self.config['use_mmap']
        if use_mmap:
            train_dataset = np.load(train_file_path, mmap_mode='r')
            val_dataset = np.load(val_file_path, mmap_mode='r')
        else:
            train_dataset = np.load(train_file_path)
            val_dataset = np.load(val_file_path)
        self.train_loader = data_loader(self.config['batch_num'], train_dataset, self.config['batch_size'], self.config['context_length'], self.device)
        self.val_loader = data_loader(self.config['batch_num'], val_dataset, self.config['batch_size'], self.config['context_length'], self.device)

    def setup_dirs(self):
        """"set up dirs to save checkpoint"""
        Path(self.config['checkpoint_dir']).mkdir(exist_ok=True)
    
    def init_model_and_optimizer(self):
        self.iteration = 1
        model = TransformerLM(
            vocab_size=self.config['model']['vocab_size'],
            context_length=self.config['model']['context_len'],
            d_model=self.config['model']['d_model'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            d_ff=self.config['model']['d_ff'],
            rope_theta=self.config['model']['rope_theta'],
            device=self.device
        )
        
        optimizer = AdamW(
            params=model.parameters(),
            lr=self.config['optim']['lr'],
            betas=tuple(self.config['optim']['betas']),
            eps=self.config['optim']['eps'],
            weight_decay=self.config['optim']['weight_decay']
        )
        
        if self.config['resume_from']:
            self.iteration = load_checkpoint(self.config['resume_from'], model, optimizer)
            
        self.model = model.to(self.device)
        self.optimzier = optimizer
        
        
    def train(self):
        self.model.count_parameters()
        pbar = tqdm(self.train_loader, total=self.config['batch_num'], desc="Training")
        for batch_idx, (input, output) in enumerate(pbar):
            prediction = self.model(input)
            loss = cross_entropy(prediction, output)
            loss.backward()
            self.optimzier.step()
            self.iteration += 1
            if (self.iteration) % self.config['checkpoint_iter'] == 0:
                save_checkpoint(self.model, self.optimzier, self.iteration, self.config['checkpoint_dir']+f"/checkpoint_{self.iteration}.ckpt")
            pbar.set_postfix({
                'epoch': f'{self.iteration}',
                'loss': f'{loss.item():.4f}'
            })
                
    def val(self):
        pass
def main():
    config = load_config('config/base.yaml')
    trainer = Trainer(config)
    trainer.train()
    trainer.val()
    
if __name__ == "__main__":
    main()