import argparse
import json
import os
import nltk
import random
from tqdm import tqdm
from shutil import copyfile


import datasets
import numpy as np
import spacy
import sklearn
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup

import logging

from utils.rouge_utils import compute_batched_sentence_rouge

logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    """HF arXiv Dataset Wrapper. It handles tokenization, max input/output seqlen, padding and batching"""
    def __init__(self, hf_arxiv_dataset, tokenizer, args):
        self.hf_arxiv_dataset = hf_arxiv_dataset
        self.tokenizer = tokenizer
        self.args = args
        self.padding = "max_length" if self.args.pad_to_max_length else False
        self.rouge = datasets.load_metric('rouge')

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.hf_arxiv_dataset)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        entry = self.hf_arxiv_dataset[idx]

        input = None
        doc_input = None
        fact_input = None
        detect_dp = True
        if "data_type" in entry.keys() and entry["data_type"] == "correct":
            detect_dp = False
        elif random.random() > 0.5:
            detect_dp = False
        else:
            pass

        if self.args.use_entire_summaries:
            original_summary_sentences = entry["original_summary_sentences"]
            orig_summ = " ".join([x for x in original_summary_sentences])
            generated_summary_sentences = entry["generated_summary_sentences"]
            gen_summ = " ".join([x for x in generated_summary_sentences])
            source_article_sentences = entry["source_article_sentences"]
            source_article = " ".join(source_article_sentences)
            input = gen_summ + " <sep> " + source_article
            targets = orig_summ
        elif self.args.use_sentence_level_summaries:
            orig_summ_sent = entry["original_summary_sent"]
            gen_summ_sent = entry["generated_summary_sent"]
            source_article = None
            if self.args.use_entire_article_as_source:
                source_article_sentences = entry["source_article_sentences"]
                source_article = " ".join(source_article_sentences)
            elif self.args.use_relevant_sents_as_source:
                relevant_indices = entry["relevant_article_sent_indices"]
                relevant_sents = [" ".join(entry["source_article_sentences"][idx-2:idx+2]) for idx in relevant_indices]
                source_article = " ".join(relevant_sents)
            else:
                print("One of x must be selected")
                exit()
            if self.args.add_full_summary_in_context:
                input = gen_summ_sent + " <sep> " + " ".join(entry["generated_summary_sentences"]) + " <sep> " + source_article
            else:
                input = gen_summ_sent + " <sep> " + source_article
            if self.args.detect_and_correct and detect_dp:
                input = "Detect" + " <sep> " + input
            elif self.args.detect_and_correct and not detect_dp:
                input = "Correct " + entry['err_span'] + " <sep> " + input
            else:
                pass

            targets = orig_summ_sent
        else:
            print("One of x must be selected")
            exit()

        labels = 1 - int(entry["label"])
        if self.args.generate_factuality_label:
            targets = str(labels) + " <sep> " + targets
        elif self.args.generate_incorrect_span:
            targets = entry['err_span'] + " <sep> " + targets
        else:
            pass

        if self.args.detect_and_correct and detect_dp:
            targets = entry['err_span']
        
        model_inputs = self.tokenizer(input, max_length=self.args.max_input_len, padding=self.padding, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.args.max_output_len, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["source_article"] = source_article

        return torch.tensor(model_inputs["input_ids"]), torch.tensor(model_inputs["labels"]), torch.tensor(idx), source_article

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """
        pad_token_id = 1
        input_ids, output_ids, indices, _ = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids, torch.stack(indices)

    def process_ood_test_example(self, entry, detect_preds=None, detect=False):
        datapoints = []
        original_summary_sentences = entry["original_summary_sentences"]
        original_summary = " ".join([x for x in original_summary_sentences])
        gen_summ = " ".join([x for x in entry["generated_summary_sentences"]])
        if self.args.use_entire_summaries:
            generated_summary_sentences = entry["generated_summary_sentences"]
            gen_summ = " ".join([x for x in generated_summary_sentences])
            source_article_sentences = entry["source_article_sentences"]
            source_article = " ".join(source_article_sentences)
            label = entry["label"]
            input = gen_summ + " <sep> " + source_article
            if self.args.detect_and_correct and detect:
                input = "Detect" + " <sep> " + input
            elif self.args.detect_and_correct and not detect:
                input = "Correct " + detect_preds[0] + " <sep> " + input
            else:
                pass
            model_inputs = self.tokenizer(input, max_length=self.args.max_input_len, padding=self.padding, truncation=True)
            datapoints.append((model_inputs, gen_summ, source_article, label))
        elif self.args.use_sentence_level_summaries:
            rouge_scores = None
            
            if self.args.use_relevant_sents_as_source:
                rouge_scores = compute_batched_sentence_rouge(entry["source_article_sentences"], entry["generated_summary_sentences"], self.rouge)
            for sent_idx, gen_summ_sent in enumerate(entry["generated_summary_sentences"]):
                source_article = None
                if self.args.use_entire_article_as_source:
                    source_article_sentences = entry["source_article_sentences"]
                    source_article = " ".join(source_article_sentences)
                elif self.args.use_relevant_sents_as_source:
                    relevant_article_sent_indices = []
                    sent_summ_art_rouge = rouge_scores[sent_idx]["rouge_scores"]
                    top_k_rouge_sents = sorted(sent_summ_art_rouge, key=lambda x: x["rougeLsum"], reverse=True)[0:3]
                    for rouge_sents in top_k_rouge_sents:
                        article_idx = rouge_sents["article_sent_idx"]
                        relevant_article_sent_indices.append(article_idx)
                    relevant_sents = [" ".join(entry["source_article_sentences"][idx-2:idx+2]) for idx in relevant_article_sent_indices]
                    source_article = " ".join(relevant_sents)
                else:
                    print("One of x must be selected")
                    exit()
                label = entry["label"]
                if self.args.add_full_summary_in_context:
                    input = gen_summ_sent + " <sep> " + " ".join(entry["generated_summary_sentences"]) + " <sep> " + source_article
                else:
                    input = gen_summ_sent + " <sep> " + source_article

                if self.args.detect_and_correct and detect:
                    input = "Detect " + " <sep> " + input
                elif self.args.detect_and_correct and not detect:
                    input = "Correct " + detect_preds[sent_idx] + " <sep> " + input
                else:
                    pass
                #print(input)
                model_inputs = self.tokenizer(input, max_length=self.args.max_input_len, padding=self.padding, truncation=True)
                datapoints.append((model_inputs, gen_summ_sent, source_article, label))

        pad_token_id = 1
        input_ids = [torch.tensor(x[0]['input_ids'], dtype=torch.int64) for x in datapoints]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        return input_ids, gen_summ, original_summary


class Summarizer(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, params):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.args = params

        # Load and update config then load a pretrained BartforConditionalGeneration
        config = AutoConfig.from_pretrained(self.args.transformer_model)
        config.gradient_checkpointing = self.args.grad_ckpt
        if self.args.resume_checkpoint_dir != "None":
            saved_model = torch.load(os.path.join(self.args.resume_checkpoint_dir, self.args.resume_checkpoint_file))
            renamed_state_dict = {}
            for k, v in saved_model["state_dict"].items():
                new_key = k.replace("model.", "")
                renamed_state_dict[new_key] = v
            self.model = AutoModelForSeq2SeqLM.from_pretrained(None, config=config, state_dict=renamed_state_dict)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.transformer_model, config=config) 

        # Load tokenizer and metric
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.transformer_model, use_fast=True)
        self.rouge = datasets.load_metric('rouge')
        self.bert_score = datasets.load_metric("bertscore")

    def forward(self, input_ids, output_ids, indices):
        """Call BartForConditionalGeneration.forward"""
        output = self.model(input_ids,
                            attention_mask=(input_ids != self.tokenizer.pad_token_id),  # mask padding tokens
                            labels=output_ids, use_cache=False)
        return output

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        outputs = self.forward(*batch)
        epoch_num = self.current_epoch+1
        self.log(f'train/train_step', batch_nb*epoch_num,
                 on_step=True, on_epoch=True)
        self.log('train/train_loss', outputs.loss, on_epoch=True)
        return {'loss': outputs.loss}

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        dataset_size = len(self.hf_dataset['train'])
        gpu_count = 1
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
        num_steps = dataset_size * self.args.epochs / gpu_count / self.args.grad_accum / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup,
                                                    num_training_steps=num_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, split_name, is_train):
        """Get training and validation dataloaders"""
        dataset_split = self.hf_dataset[split_name]
        dataset = SummarizationDataset(hf_arxiv_dataset=dataset_split, tokenizer=self.tokenizer, args=self.args)
        sampler = None
        if torch.cuda.is_available() and is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
        else:
            sampler = None
        print(sampler, is_train, (is_train and sampler is None))
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(is_train and sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

    def train_dataloader(self):
        return self._get_dataloader('train', is_train=True)

    def val_dataloader(self):
        return self._get_dataloader('validation', is_train=False)

    def test_dataloader(self):
        return self._get_dataloader('test', is_train=False)

    def _evaluation_step(self, split, batch, batch_nb):
        """Validaton or Testing - predict output, compare it with gold, compute rouge1, 2, L, and log result"""
        # Generate
        input_ids, output_ids, indices = batch
        outputs = self.model.generate(input_ids=input_ids,
                                      attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                      use_cache=True, max_length=self.args.max_output_len, num_beams=1,
                                      return_dict_in_generate=True, output_hidden_states=True)
        generated_ids = outputs["sequences"]

        # Convert predicted and gold token ids to strings
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

        if self.args.generate_factuality_label:
            pred_labels, orig_labels = [], []
            preds, origs = [], []
            for pred, summ in zip(predictions, references):
                if "<sep>" in pred and (pred.split("<sep>")[0].strip() == '0' or pred.split("<sep>")[0].strip() == '1'):
                    pred_label, pred_text = pred.split("<sep>")
                    orig_label, orig_text = summ.split("<sep>")
                    pred_labels.append(int(pred_label.strip()))
                    preds.append(pred_text.strip())
                    orig_labels.append(int(orig_label.strip()))
                    origs.append(orig_text.strip())
            if len(pred_labels) > 0:
                label_f1 = sklearn.metrics.f1_score(orig_labels, pred_labels)
                outputs[f'{split}_fact_label_f1'] = input_ids.new_zeros(1) + label_f1
            predictions = preds
            references = origs
        if args.generate_incorrect_span:
            pred_labels, orig_labels = [], []
            preds, origs = [], []
            for pred, summ in zip(predictions, references):
                if "<sep>" in pred:
                    pred_label, pred_text = pred.split("<sep>")
                    orig_label, orig_text = summ.split("<sep>")
                    pred_labels.append(int(pred_label.strip()==orig_label.strip()))
                    preds.append(pred_text.strip())
                    orig_labels.append(int(1))
                    origs.append(orig_text.strip())
            predictions = preds
            references = origs
       
        preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
        summary = ["\n".join(nltk.sent_tokenize(summ.strip())) for summ in references]

        # Compute rouge
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        results = self.rouge.compute(predictions=preds, references=summary)
        bert_metric_names = ['precision', 'recall', 'f1']
        bertscore_results = self.bert_score.compute(predictions=preds, references=summary, lang='en')
        outputs = {}
        for metric_name in metric_names:
            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            outputs[f'{split}_{metric_name}'] = metric_val
        for metric_name in bert_metric_names:
            metric_val = input_ids.new_zeros(1) + sum(bertscore_results[metric_name])/len(bertscore_results[metric_name])
            outputs[f'{split}_bertscore_{metric_name}'] = metric_val
        ref_preds = []
        for idx, pred, ref in zip(indices, predictions, references):
            ref_preds.append({"idx": idx.cpu().item(), "pred": pred, "ref": ref})
        outputs["predictions"] = ref_preds
        return outputs

    def _evaluation_epoch_end(self, split, step_outputs):
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 'fact_label_f1']
        aggregated_metrics = {}
        for metric_name in metric_names:
            aggregated_metrics[f'{split}_{metric_name}'] = []

        predictions = []
        for pred in step_outputs:
            predictions.extend(pred["predictions"])
            del pred["predictions"]
            for key, value in pred.items():
                aggregated_metrics[key].append(value)

        for key, value in aggregated_metrics.items():
            if len(value) != 0:
                aggregated_metrics[key] = torch.mean(torch.stack(value, dim=0), dim=0, keepdim=False)
                self.log(f'{split}/{key}_epoch', aggregated_metrics[key], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        aggregated_metrics["predictions"] = predictions
        return aggregated_metrics

    def validation_step(self, batch, batch_nb):
        """Validation - predict output, compare it with gold, compute rouge1, and return result"""
        epoch_num = self.current_epoch+1
        return self._evaluation_step('val', batch, batch_nb)

    def validation_epoch_end(self, validation_step_outputs):
        fp = open(args.output_dir+"/val_metrics.txt", "a+")
        aggregated_metrics = self._evaluation_epoch_end('val', validation_step_outputs)
        for key, value in aggregated_metrics.items():
            fp.write(f'{key}_epoch: '+str(aggregated_metrics[key])+"\n")

    def test_step(self, batch, batch_nb):
        """Test - predict output, compare it with gold, compute rouge1, and return result"""
        return self._evaluation_step('test', batch, batch_nb)

    def test_epoch_end(self, test_step_outputs):
        aggregated_metrics = self._evaluation_epoch_end('test', test_step_outputs)
        for key, value in aggregated_metrics.items():
            if key != 'predictions':
                if len(value) != 0:
                    aggregated_metrics[key] = value.cpu().item()
        fp = open(args.output_dir+"/metrics.json", "w")
        fp.write(json.dumps(aggregated_metrics))
        return aggregated_metrics

    @staticmethod
    def add_model_specific_args(parser):
        # **************** Parameters that we will NOT change during this tutorial **************** #
        parser.add_argument("--name", type=str, default='test', help="Name of expt")
        parser.add_argument("--use_entire_summaries", type=bool, default=False,
                            help="Correct entire multi sentence summary")
        parser.add_argument("--use_sentence_level_summaries", type=bool, default=False,
                            help="Use single sentence of summary for correction")
        parser.add_argument("--use_entire_article_as_source", type=bool, default=False,
                            help="Use entire document as context")
        parser.add_argument("--use_relevant_sents_as_source", type=bool, default=False,
                            help="Use only relevant sents to summary sents from article as context")
        parser.add_argument("--add_full_summary_in_context", type=bool, default=False,
                            help="Add full summary as additional context")
        parser.add_argument("--generate_factuality_label", type=bool, default=False,
                            help="Add factuality label when generating")
        parser.add_argument("--generate_incorrect_span", type=bool, default=False,
                            help="Add incorrect span  when generating")
        parser.add_argument("--detect_and_correct", type=bool, default=False,
                            help="Detect and correct span according to prompt")
        parser.add_argument("--filter_using_factcc", type=bool, default=False,
                            help="Do not correct if factcc factual")

        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--lr", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--pad_to_max_length", type=bool, default=False,
                            help="Whether to pad all samples to model maximum sentence length. ")
        parser.add_argument("--limit_val_batches", default=0.001, type=float, help='Percent of validation data used')
        parser.add_argument("--limit_test_batches", default=0.002, type=float, help='Percent of test data used')
        parser.add_argument("--limit_train_batches", default=0.001, type=float, help='Percent of training data used')
        parser.add_argument("--max_output_len", type=int, default=256, help="maximum num of wordpieces in the summary")
        parser.add_argument("--transformer_model", type=str, default='facebook/bart-base', help="transformer_model tyoe")
        parser.add_argument("--output_dir", type=str, default='./saved_models/test', help="Location of output dir")
        parser.add_argument("--resume_checkpoint_dir", type=str, default="None", help="Location of resume ckpt")
        parser.add_argument("--resume_checkpoint_file", type=str, default="None", help="Filename of resume ckpt")
        parser.add_argument("--data_dir", type=str, default='./data/', help="Location of input dir")
        parser.add_argument("--ood_test_datapath", type=str, default='./data/bart_test.json', help="Location of input dir")
        parser.add_argument("--baseline_fcc_file", type=str, default='/remote/bones/user/vbalacha/summary_fact_corrector/output_dir/bart_large_cnn/factcc_corr/factcc_eval_predictions_wcnnid.jsonl', help="Location of FactCC Results of Baseline")
        parser.add_argument("--val_every", default=0.33, type=float, help='Validation every')
        parser.add_argument("--num_beams", default=1, type=int, help='Num beams for inference')
        parser.add_argument("--do_train", type=bool, default=False,
                            help="Do Train")
        parser.add_argument("--do_predict", type=bool, default=False,
                            help="Do Predict")
        parser.add_argument("--do_predict_ood", type=bool, default=False,
                            help="Do Predict OOD")

# **************** Parameters that we will change during this tutorial **************** #
        parser.add_argument("--max_input_len", type=int, default=1024, help="maximum num of wordpieces in the input")
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')

        return parser


if __name__ == "__main__":
    # Setup command line args
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = Summarizer.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    # Init a PL module
    set_seed(args.seed)
    summarizer = Summarizer(args)

    # Load the arXiv dataset from local
    summarizer.hf_dataset = datasets.load_dataset('json', data_files={"train": os.path.join(args.data_dir, "train.json"),
                                                                      "validation": os.path.join(args.data_dir, "validation.json"),
                                                                      "test": os.path.join(args.data_dir, "test.json")})
                                                                      #"ood_test": args.ood_test_datapath})

    checkpoint_callback = ModelCheckpoint(monitor='val/val_rougeLsum_epoch',
                                          dirpath=args.output_dir,
                                          filename='{epoch:02d}-{step}-val_rougeLsum_epoch{val/val_rougeLsum_epoch:.4f}',
                                          save_top_k=5,
                                          mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger(name=args.name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ckpt_path = None
    if args.resume_checkpoint_dir != "None":
        ckpt_path = os.path.join(args.resume_checkpoint_dir, args.resume_checkpoint_file)

    # Construct a PL trainer
    if torch.cuda.is_available():
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
                             callbacks=[checkpoint_callback, lr_monitor],
                             val_check_interval=args.val_every,
                             resume_from_checkpoint=ckpt_path,
                             track_grad_norm=2,
                             logger=wandb_logger)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs,
                             replace_sampler_ddp=False,
                             num_sanity_val_steps=0,
                             default_root_dir=args.output_dir,
                             limit_val_batches=args.limit_val_batches,
                             limit_train_batches=args.limit_train_batches,
                             limit_test_batches=args.limit_test_batches,
                             accumulate_grad_batches=args.grad_accum,
                             callbacks=[checkpoint_callback, lr_monitor],
                             val_check_interval=args.val_every,
                             resume_from_checkpoint=ckpt_path,
                             track_grad_norm=2,
                             logger=wandb_logger)

    if not args.do_train and not args.do_predict and not args.do_predict_ood:
        print("One of --do_train, --do_predict, --do_predict_ood should be given")
        exit()

    if args.do_train:
        # Start training
        print("Training")
        trainer.fit(summarizer)
        pa = checkpoint_callback.best_model_path
        copyfile(pa, os.path.join(args.output_dir, 'best.ckpt'))

    if args.do_predict:
        # Start testing
        print("Test on in-dist-domain data")
        result = trainer.test(summarizer)
        copyfile(os.path.join(args.output_dir, 'metrics.json'), os.path.join(args.output_dir, 'test_metrics.json'))

    if args.do_predict_ood:
        print("Test on out-dist-domain data")
        split = 'ood_test'
        summarizer.hf_dataset = datasets.load_dataset('json', data_files={"ood_test": args.ood_test_datapath})
        dataset_split = summarizer.hf_dataset['ood_test']
        orig_test_dataset = SummarizationDataset(hf_arxiv_dataset=dataset_split, tokenizer=summarizer.tokenizer, args=summarizer.args)
        output_test_preds_file = os.path.join(args.output_dir, "test_generations.json")
        fcc_test_file = os.path.join(args.output_dir, "data-dev.jsonl")
        dae_test_file = os.path.join(args.output_dir, "dae_input.txt")
        
        factcc_res = {}
        if args.filter_using_factcc:
            print('Using FactCC to filter already factual')
            baseline_factcc_file = open(args.baseline_fcc_file)
            for line in baseline_factcc_file:
                item = json.loads(line)
                if item['xsum_id'] not in factcc_res:
                    factcc_res[item['xsum_id']] = 0
                if int(item['pred']) == 1:
                    factcc_res[item['xsum_id']] = 1
                # if item['data_id'] not in factcc_res:
                #     factcc_res[int(item['data_id'])] = 0
                # if int(item['pred']) == 1:
                #     factcc_res[int(item['data_id'])] = 1
        
        result_aggregator = []

        if torch.cuda.is_available():
            summarizer.model = summarizer.model.to(device=torch.device('cuda'))
        with open(output_test_preds_file, "w") as writer, open(fcc_test_file, "w") as fccp, open(dae_test_file, "w") as daep:
            for idx, entry in tqdm(enumerate(dataset_split)):
                # if idx >= 10:
                #     break
                source_article_sentences = entry["source_article_sentences"]
                if len(source_article_sentences) == 0:
                    print(idx)
                    continue
                source_article = " ".join(source_article_sentences)
                predictions = None
                detect_predictions = None
                if args.detect_and_correct:
                    gensumm_sent_input_ids, gen_summ, original_summary = orig_test_dataset.process_ood_test_example(entry, detect=True)

                    gensumm_sent_input_ids = gensumm_sent_input_ids.to(summarizer.model.device)
                    outputs = summarizer.model.generate(input_ids=gensumm_sent_input_ids,
                                                        attention_mask=(gensumm_sent_input_ids != summarizer.tokenizer.pad_token_id),
                                                        use_cache=False, max_length=args.max_output_len, num_beams=arg.num_beams,
                                                        #use_cache=False, max_length=args.max_output_len, num_beams=5,
                                                        return_dict_in_generate=True, output_hidden_states=True)
                    generated_ids = outputs["sequences"]
                    detect_predictions = summarizer.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)

                    gensumm_sent_input_ids, gen_summ, original_summary = orig_test_dataset.process_ood_test_example(entry, detect_preds=detect_predictions, detect=False)

                    gensumm_sent_input_ids = gensumm_sent_input_ids.to(summarizer.model.device)
                    outputs = summarizer.model.generate(input_ids=gensumm_sent_input_ids,
                                                        attention_mask=(gensumm_sent_input_ids != summarizer.tokenizer.pad_token_id),
                                                        use_cache=False, max_length=args.max_output_len, num_beams=args.num_beams,
                                                        #use_cache=False, max_length=args.max_output_len, num_beams=5,
                                                        return_dict_in_generate=True, output_hidden_states=True)
                    generated_ids = outputs["sequences"]

                    # Convert predicted and gold token ids to strings
                    predictions = summarizer.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)

                else:
                    gensumm_sent_input_ids, gen_summ, original_summary = orig_test_dataset.process_ood_test_example(entry)

                    gensumm_sent_input_ids = gensumm_sent_input_ids.to(summarizer.model.device)
                    outputs = summarizer.model.generate(input_ids=gensumm_sent_input_ids,
                                                  attention_mask=(gensumm_sent_input_ids != summarizer.tokenizer.pad_token_id),
                                                  use_cache=False, max_length=args.max_output_len, num_beams=args.num_beams,
                                                  length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True,
                                                  #use_cache=False, max_length=args.max_output_len, num_beams=5,
                                                  return_dict_in_generate=True, output_hidden_states=True)
                    generated_ids = outputs["sequences"]

                    # Convert predicted and gold token ids to strings
                    predictions = summarizer.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
                #print(predictions)
                outputs = {}
                if args.generate_factuality_label:
                    sent_pred_labels = []
                    summ_pred_labels = []
                    preds = []
                    for pred in predictions:
                        if "<sep>" in pred and (pred.split("<sep>")[0].strip() == '0' or pred.split("<sep>")[0].strip() == '1'):
                            pred_label, pred_text = pred.split("<sep>")
                            sent_pred_labels.append(int(pred_label.strip()))
                            preds.append(pred_text.strip())
                    if len(sent_pred_labels) > 0:
                        outputs[f'{split}_fact_incorrect'] = sum(sent_pred_labels)
                        outputs[f'{split}_fact_correct'] = len(sent_pred_labels) - sum(sent_pred_labels)
                        outputs[f'{split}_fact_num'] = len(sent_pred_labels)
                        if args.use_sentence_level_summaries:
                            outputs[f'{split}_fact_summ_incorrect'] = 1 if 1 in sent_pred_labels else 0
                    predictions = preds
                if args.generate_incorrect_span:
                    preds = []
                    spans = []
                    for pred in predictions:
                        if "<sep>" in pred:
                            pred_span, pred_text = pred.split("<sep>")
                            preds.append(pred_text.strip())
                            spans.append(pred_span)
                    predictions = preds

                if args.use_entire_summaries:
                    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
                elif args.use_sentence_level_summaries:
                    preds = ["\n".join([pred.strip() for pred in predictions])]
                else:
                    print("One of x must be selected")
                    exit()

                if args.filter_using_factcc:
                    #if factcc_res[idx] == 0: # IF correct use orig generated summ
                    if factcc_res[entry["xsum_id"]] == 0: # IF correct use orig generated summ
                        print('Already Factual - Using original Summary')
                        preds = ["\n".join(nltk.sent_tokenize(gen_summ.strip()))]
                        predictions = [p.strip() for p in nltk.sent_tokenize(gen_summ.strip())]

                summary = ["\n".join(nltk.sent_tokenize(original_summary.strip()))]

                # Compute rouge
                metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
                results = summarizer.rouge.compute(predictions=preds, references=summary)
                bert_metric_names = ['precision', 'recall', 'f1']
                bertscore_results = summarizer.bert_score.compute(predictions=preds, references=summary, lang='en')
                for metric_name in metric_names:
                    metric_val = results[metric_name].mid.fmeasure
                    outputs[f'{split}_{metric_name}'] = metric_val
                for metric_name in bert_metric_names:
                    metric_val = sum(bertscore_results[metric_name])/len(bertscore_results[metric_name])
                    outputs[f'{split}_bertscore_{metric_name}'] = metric_val
                
                outputs["predictions"] = {"idx": idx, "pred": " ".join([pred.strip("\n") for pred in predictions]), "ref": original_summary, "old_summ":gen_summ}
                if detect_predictions is not None:
                    outputs["predictions"]["detect_predictions"] = str(detect_predictions)
                if args.generate_incorrect_span:
                    outputs["predictions"]["err_span"] = str(spans)
                result_aggregator.append(outputs)

                for summary_sent in predictions:
                    summary_sent = summary_sent.strip("\n")
                    if summary_sent == "":
                        continue
                    datum = {"text": source_article,
                             "claim": summary_sent,
                             "label": "CORRECT"}
                    fccp.write(json.dumps(datum)+"\n")
                daep.write(str(source_article).replace("\n", " ")+"\n"+str(" ".join([pred.strip("\n") for pred in predictions])).replace("\n", " ")+"\n\n")
                writer.write(json.dumps(outputs)+"\n")

            metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
            aggregated_metrics = {}
            for metric_name in metric_names:
                aggregated_metrics[f'{split}_{metric_name}'] = []
            if args.generate_factuality_label:
                aggregated_metrics[f'{split}_fact_incorrect'] = []
                aggregated_metrics[f'{split}_fact_correct'] = []
                aggregated_metrics[f'{split}_fact_num'] = []
                if args.use_sentence_level_summaries:
                    aggregated_metrics[f'{split}_fact_summ_incorrect'] = []
            
            predictions = []
            for pred in result_aggregator:
                predictions.extend(pred["predictions"])
                del pred["predictions"]
                for key, value in pred.items():
                    aggregated_metrics[key].append(value)

            for key in metric_names:
                value = aggregated_metrics[f'{split}_{key}']
                aggregated_metrics[f'{split}_{key}'] = sum(value)/len(value)

            if args.generate_factuality_label:
                aggregated_metrics[f'{split}_fact_incorrect'] = sum(aggregated_metrics[f'{split}_fact_incorrect'])
                aggregated_metrics[f'{split}_fact_correct'] = sum(aggregated_metrics[f'{split}_fact_correct'])
                aggregated_metrics[f'{split}_fact_num'] = sum(aggregated_metrics[f'{split}_fact_num'])
                if args.use_sentence_level_summaries:
                    aggregated_metrics[f'{split}_fact_summ_incorrect'] = sum(aggregated_metrics[f'{split}_fact_summ_incorrect'])/len(aggregated_metrics[f'{split}_fact_summ_incorrect'])

            print(aggregated_metrics)
            fp = open(args.output_dir+"/metrics.json", "w")
            fp.write(json.dumps(aggregated_metrics))
            fp.close()
            copyfile(os.path.join(args.output_dir, 'metrics.json'), os.path.join(args.output_dir, 'ood_test_metrics.json'))

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
