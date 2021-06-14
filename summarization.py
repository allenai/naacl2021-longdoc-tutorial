import argparse
import json
import os

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup


class SummarizationDataset(Dataset):
    """HF arXiv Dataset Wrapper. It handles tokenization, max input/output seqlen, padding and batching"""

    def __init__(self, hf_arxiv_dataset, tokenizer, args):
        self.hf_arxiv_dataset = hf_arxiv_dataset
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.hf_arxiv_dataset)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        entry = self.hf_arxiv_dataset[idx]
        input_ids = self.tokenizer.encode(entry['article'], truncation=True, max_length=self.args.max_input_len,
                                          padding='max_length')  # padding to max seqlen for const memory/example
        output_ids = self.tokenizer.encode(entry['abstract'], truncation=True, max_length=self.args.max_output_len,
                                          padding='max_length')  # padding to max seqlen for const memory/example
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids


class Summarizer(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.args = params

        # Load and update config then load a pretrained LEDForConditionalGeneration
        config = AutoConfig.from_pretrained('allenai/led-base-16384')
        config.gradient_checkpointing = self.args.grad_ckpt
        config.attention_window = [self.args.attention_window] * len(config.attention_window)
        self.model = AutoModelForSeq2SeqLM.from_pretrained('allenai/led-base-16384', config=config)

        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/led-base-16384', use_fast=True)
        self.rouge = datasets.load_metric('rouge')

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""

        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Gradient Accumulation caveat 1:
        # For gradient accumulation to work, all model parameters should contribute
        # to the computation of the loss. Remember that the self-attention layers in the LED model
        # have two sets of qkv layers, one for local attention and another for global attention.
        # If we don't use any global attention, the global qkv layers won't be used and
        # PyTorch will throw an error. This is just a PyTorch implementation limitation
        # not a conceptual one (PyTorch 1.8.1).
        # The following line puts global attention on the <s> token to make sure all model
        # parameters which is necessery for gradient accumulation to work.
        global_attention_mask[:, 0] = 1

        # # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1

        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        return global_attention_mask

    def forward(self, input_ids, output_ids):
        """Call LEDForConditionalGeneration.forward"""
        return self.model(input_ids,
                          attention_mask=(input_ids != self.tokenizer.pad_token_id),  # mask padding tokens
                          global_attention_mask=self._set_global_attention_mask(input_ids),  # set global attention
                          labels=output_ids, use_cache=False)

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        outputs = self.forward(*batch)
        return {'loss': outputs.loss}

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        dataset_size = len(self.hf_dataset['train'])
        gpu_count = torch.cuda.device_count()
        num_steps = dataset_size * self.args.epochs / gpu_count / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup,
                                                    num_training_steps=num_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, split_name, is_train):
        """Get training and validation dataloaders"""
        dataset_split = self.hf_dataset[split_name]
        dataset = SummarizationDataset(hf_arxiv_dataset=dataset_split, tokenizer=self.tokenizer, args=self.args)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

    def train_dataloader(self):
        return self._get_dataloader('train', is_train=True)

    def val_dataloader(self):
        return self._get_dataloader('validation', is_train=False)

    def test_dataloader(self):
        return self._get_dataloader('test', is_train=False)

    def validation_step(self, batch, batch_nb):
        """Validation - predict output, compare it with gold, compute rouge1, and return result"""
        # Generate
        input_ids, output_ids = batch
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            global_attention_mask=self._set_global_attention_mask(input_ids),
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        # Compute rouge
        results = self.rouge.compute(predictions=predictions, references=references)
        rouge1 = input_ids.new_zeros(1) + results["rouge1"].mid.fmeasure
        rouge2 = input_ids.new_zeros(1) + results["rouge2"].mid.fmeasure
        rougel = input_ids.new_zeros(1) + results["rougeL"].mid.fmeasure
        rougelsum = input_ids.new_zeros(1) + results["rougeLsum"].mid.fmeasure

        # Log metric
        self.log('val_rouge1', rouge1, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_rouge2', rouge2, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_rougeL', rougel, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('val_rougeLsum', rougelsum, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        metrics = {"val_rouge1": rouge1,
                   "val_rouge2": rouge2,
                   "val_rougeL": rougel,
                   "val_rougeLsum": rougelsum}
        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        aggregated_metrics = {"val_rouge1": [],
                              "val_rouge2": [],
                              "val_rougeL": [],
                              "val_rougeLsum": []}
        for pred in validation_step_outputs:
            for key, value in pred.items():
                aggregated_metrics[key].append(value.cpu().item())

        for key, value in aggregated_metrics.items():
            aggregated_metrics[key] = np.mean(value)

        self.log("Val Rouge 1: ", aggregated_metrics["val_rouge1"])
        self.log("Val Rouge 2: ", aggregated_metrics["val_rouge2"])
        self.log("Val Rouge L: ", aggregated_metrics["val_rougeL"])
        self.log("Val Rouge Lsum: ", aggregated_metrics["val_rougeLsum"])

        fp = open(args.output_dir+"/val_metrics.txt", "a+")
        fp.write("Val Rouge 1: "+str(aggregated_metrics["val_rouge1"])+"\n")
        fp.write("Val Rouge 2: "+str(aggregated_metrics["val_rouge2"])+"\n")
        fp.write("Val Rouge L: "+str(aggregated_metrics["val_rougeL"])+"\n")
        fp.write("Val Rouge Lsum: "+str(aggregated_metrics["val_rougeLsum"])+"\n\n")
        fp.close()

    def test_step(self, batch, batch_nb):
        """Test - predict output, compare it with gold, compute rouge1, and return result"""
        # Generate
        input_ids, output_ids = batch
        generated_ids = self.model.generate(input_ids=input_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            global_attention_mask=self._set_global_attention_mask(input_ids),
                                            use_cache=True, max_length=self.args.max_output_len, num_beams=1)

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        # Compute rouge
        results = self.rouge.compute(predictions=predictions, references=references)
        rouge1 = input_ids.new_zeros(1) + results["rouge1"].mid.fmeasure
        rouge2 = input_ids.new_zeros(1) + results["rouge2"].mid.fmeasure
        rougel = input_ids.new_zeros(1) + results["rougeL"].mid.fmeasure
        rougelsum = input_ids.new_zeros(1) + results["rougeLsum"].mid.fmeasure

        # Log metric
        self.log('test_rouge1', rouge1, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('test_rouge2', rouge2, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('test_rougel', rougel, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('test_rougelsum', rougel, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        metrics = {"test_rouge1": rouge1,
                   "test_rouge2": rouge2,
                   "test_rougeL": rougel,
                   "test_rougeLsum": rougelsum}
        return metrics

    def test_epoch_end(self, test_step_outputs):
        aggregated_metrics = {"test_rouge1": [],
                              "test_rouge2": [],
                              "test_rougeL": [],
                              "test_rougeLsum": []}

        for pred in test_step_outputs:
            for key, value in pred.items():
                aggregated_metrics[key].append(value.cpu().item())

        for key, value in aggregated_metrics.items():
            aggregated_metrics[key] = np.mean(value)

        self.log("Test Rouge 1: ", aggregated_metrics["test_rouge1"])
        self.log("Test Rouge 2: ", aggregated_metrics["test_rouge2"])
        self.log("Test Rouge L: ", aggregated_metrics["test_rougeL"])
        self.log("Test Rouge Lsum: ", aggregated_metrics["test_rougeLsum"])

        fp = open(args.output_dir+"/metrics.json", "w")
        fp.write(json.dumps(aggregated_metrics))

    @staticmethod
    def add_model_specific_args(parser):
        # **************** Parameters that we will NOT change during this tutorial **************** #
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--limit_val_batches", default=0.005, type=float, help='Percent of validation data used')
        parser.add_argument("--limit_test_batches", default=0.005, type=float, help='Percent of test data used')
        parser.add_argument("--limit_train_batches", default=0.002, type=float, help='Percent of training data used')
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")
        parser.add_argument("--output_dir", type=str, default='./saved_models/test', help="Location of output dir")
        parser.add_argument("--val_every", default=0.33, type=float, help='Validation every')

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
    parser = Summarizer.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    # Init a PL module
    set_seed(args.seed)
    summarizer = Summarizer(args)

    # Load the arXiv dataset from HF datasets
    summarizer.hf_dataset = datasets.load_dataset('scientific_papers', 'arxiv')

    checkpoint_callback = ModelCheckpoint(monitor='val_rouge1',
                                          dirpath=args.output_dir,
                                          save_top_k=3)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Construct a PL trainer
    trainer = pl.Trainer(gpus=-1,
                         accelerator='ddp',
                         # Gradient Accumulation caveat 2:
                         # For gradient accumulation to work with DistributedDataParallel,
                         # the `find_unused_parameters` should be `False`. Without it,
                         # you get a not-very-helpful error message (PyTorch 1.8.1)
                         plugins=[pl.plugins.ddp_plugin.DDPPlugin(find_unused_parameters=False)],
                         max_epochs=args.epochs,
                         replace_sampler_ddp=False,
                         num_sanity_val_steps=0,
                         default_root_dir=args.output_dir,
                         limit_val_batches=args.limit_val_batches,
                         limit_train_batches=args.limit_train_batches,
                         limit_test_batches=args.limit_test_batches,
                         precision=16 if args.fp16 else 32,
                         accumulate_grad_batches=args.grad_accum,
                         callbacks=[checkpoint_callback],
                         val_check_interval=args.val_every
                         )
    # Start training
    trainer.fit(summarizer)

    # Start testing
    result = trainer.test()
    print(result)

'''
conda create --name tutorial python=3.7
conda activate tutorial

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
git clone git@github.com:allenai/naacl2021-longdoc-tutorial.git
cd naacl2021-longdoc-tutorial
pip install -r requirements.txt
PYTHONWARNINGS="ignore" CUDA_VISIBLE_DEVICES=6,7   python summarization.py  \
    --fp16  --batch_size 2  --grad_accum 1 --grad_ckpt   \
    --max_input_len  16384 --attention_window  1024
'''
