import os
import argparse
import datasets
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup


class SummarizationDataset(Dataset):
    """HF Dataset Wrapper. It handles tokenization, max input/output seqlen, padding and batching"""
    def __init__(self, hf_dataset, tokenizer, args):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        entry = self.hf_dataset[idx]
        input_ids = self.tokenizer.encode(entry['article'], truncation=True, max_length=self.args.max_input_len)
        output_ids = self.tokenizer.encode(entry['abstract'], truncation=True, max_length=self.args.max_output_len)
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids


class Summarizer(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, params):
        super().__init__()
        self.args = params

        # Load the config, update the attention window size, then load the LEDForConditionalGeneration model
        config = AutoConfig.from_pretrained('allenai/led-base-16384')
        config.attention_window = [self.args.attention_window] * len(config.attention_window)
        self.model = AutoModelForSeq2SeqLM.from_pretrained('allenai/led-base-16384', config=config)

        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/led-base-16384', use_fast=True)
        self.rouge = datasets.load_metric('rouge')

    def configure_optimizers(self):
        '''Configure the optimizer and the learning rate scheduler'''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        num_steps = len(self.hf_dataset['train']) * self.args.epochs / torch.cuda.device_count() / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, split_name, is_train):
        '''Get training and validation dataloaders'''
        dataset = SummarizationDataset(hf_dataset=self.hf_dataset[split_name], tokenizer=self.tokenizer, args=self.args)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

    def train_dataloader(self):
        return self._get_dataloader('train', is_train=True)

    def val_dataloader(self):
        return self._get_dataloader('validation', is_train=False)

    def _prepare_led_global_attention_mask(self, input_ids):
        '''Configure the global attention pattern based on the task'''

        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        # global_attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        # global_attention_mask[:, 0] = 2  # global attention on one token for all model params to be used, which is important for gradient checkpointing to work
        return global_attention_mask

    def forward(self, input_ids, output_ids):
        # Prepare global attention pattern
        global_attention_mask = self._prepare_led_global_attention_mask(input_ids)

        # Call LEDForConditionalGeneration.forward
        return self.model(input_ids, global_attention_mask=global_attention_mask, labels=output_ids)

    def training_step(self, batch, batch_nb):
        # Call the forward pass then return loss
        outputs = self.forward(*batch)
        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        # Disable gradients to save memory in the forward pass
        for p in self.model.parameters():
            p.requires_grad = False

        # Generate
        input_ids, output_ids = batch
        global_attention_mask = self._prepare_led_global_attention_mask(input_ids)
        generated_ids = self.model.generate(input_ids=input_ids, global_attention_mask=global_attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        # Compute rouge
        results = self.rouge.compute(predictions=predictions, references=references)
        rouge1 = input_ids.new_zeros(1) + results["rouge1"].mid.fmeasure

        # Log metric
        self.log('val_rouge1', rouge1, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def validation_epoch_end(self, outputs):
        # Enable gradients again when validation is done
        for p in self.model.parameters():
            p.requires_grad = True

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        # **************** Parameters that we will NOT change during this tutorial **************** #
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--limit_val_batches", default=0.005, type=float, help='Percent of validation data used')
        parser.add_argument("--limit_train_batches", default=0.002, type=float, help='Percent of training data used')
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")

        # **************** Parameters that we will change during this tutorial **************** #
        parser.add_argument("--max_input_len", type=int, default=8192, help="maximum num of wordpieces in the input")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_window", type=int, default=1024, help="Attention window")

        return parser


if __name__ == "__main__":
    # Setup command line args
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = Summarizer.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()

    # Init a PL module
    set_seed(args.seed)
    summarizer = Summarizer(args)

    # Load the arXiv dataset from HF datasets
    summarizer.hf_dataset = datasets.load_dataset('scientific_papers', 'arxiv')

    # Construct a PL trainer
    trainer = pl.Trainer(gpus=-1, distributed_backend='ddp',
                         max_epochs=args.epochs,
                         replace_sampler_ddp=False,
                         num_sanity_val_steps=0,
                         limit_val_batches=args.limit_val_batches,
                         limit_train_batches=args.limit_train_batches,
                         precision=16 if args.fp16 else 32,
                         accumulate_grad_batches=args.grad_accum,
                         )
    # Start training
    trainer.fit(summarizer)

'''
conda create --name tutorial python=3.7
conda activate tutorial

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
git clone git@github.com:allenai/naacl2021-longdoc-tutorial.git
cd naacl2021-longdoc-tutorial
pip install -r requirements.txt
PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=0,1   python summarization.py

PYTHONWARNINGS="ignore" srun --gpus=2  -w  allennlp-server3   python summarization.py
'''
