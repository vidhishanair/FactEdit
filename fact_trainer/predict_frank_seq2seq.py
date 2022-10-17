import argparse
import json
import os
import nltk
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

from fact_trainer.train_seq2seq import Summarizer, SummarizationDataset

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Setup command line args
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = Summarizer.add_model_specific_args(main_arg_parser)
    args = parser.parse_args()

    # Init a PL module
    set_seed(args.seed)
    summarizer = Summarizer(args)

    # Load the arXiv dataset from local
    summarizer.hf_dataset = datasets.load_dataset('json', data_files={"ood_test": args.ood_test_datapath})

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
                             val_check_interval=args.val_every,
                             resume_from_checkpoint=ckpt_path,
                             track_grad_norm=2)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs,
                             replace_sampler_ddp=False,
                             num_sanity_val_steps=0,
                             default_root_dir=args.output_dir,
                             limit_val_batches=args.limit_val_batches,
                             limit_train_batches=args.limit_train_batches,
                             limit_test_batches=args.limit_test_batches,
                             accumulate_grad_batches=args.grad_accum,
                             val_check_interval=args.val_every,
                             resume_from_checkpoint=ckpt_path,
                             track_grad_norm=2)

    if not args.do_predict_ood:
        print("One of --do_train, --do_predict, --do_predict_ood should be given")
        exit()

    if args.do_predict_ood:
        print("Test on out-dist-domain data")
        split = 'ood_test'
        summarizer.hf_dataset = datasets.load_dataset('json', data_files={"ood_test": args.ood_test_datapath})
        dataset_split = summarizer.hf_dataset['ood_test']
        orig_test_dataset = SummarizationDataset(hf_arxiv_dataset=dataset_split, tokenizer=summarizer.tokenizer, args=summarizer.args)
        output_test_preds_file = os.path.join(args.output_dir, "frank_test_generations.json")
        fcc_test_file = os.path.join(args.output_dir, "data-dev.jsonl")
        dae_test_file = os.path.join(args.output_dir, "dae_input.txt")
        dae_test_file_adj = os.path.join(args.output_dir, "dae_input_adj.txt")
        result_aggregator = []

        frank_data = json.load(open('data/baseline_factuality_metrics_outputs.json'))
        frank_p_data = {'cnn': {'test': {}, 'valid': {}},
                        'xsum': {'test': {}, 'valid': {}}}
        for data_point in frank_data:
            frank_p_data[data_point['hash']] = data_point

        if torch.cuda.is_available():
            summarizer.model = summarizer.model.to(device=torch.device('cuda'))
        with open(output_test_preds_file, "w") as writer, open(fcc_test_file, "w") as fccp, open(dae_test_file, "w") as daep, open(dae_test_file_adj, 'w') as daep_adj:
            for idx, entry in tqdm(enumerate(dataset_split)):
                #if idx >= 10:
                #    break
                source_article_sentences = entry["source_article_sentences"]
                if len(source_article_sentences) == 0:
                    print(idx)
                    continue
                source_article = " ".join(source_article_sentences)
                gensumm_sent_input_ids, gen_summ, original_summary = orig_test_dataset.process_ood_test_example(entry)

                gensumm_sent_input_ids = gensumm_sent_input_ids.to(summarizer.model.device)
                outputs = summarizer.model.generate(input_ids=gensumm_sent_input_ids,
                                                    attention_mask=(gensumm_sent_input_ids != summarizer.tokenizer.pad_token_id),
                                                    use_cache=False, max_length=args.max_output_len, num_beams=1,
                                                    #use_cache=False, max_length=args.max_output_len, num_beams=5,
                                                    return_dict_in_generate=True, output_hidden_states=True)
                generated_ids = outputs["sequences"]

                # Convert predicted and gold token ids to strings
                predictions = summarizer.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
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
                    for pred_idx, pred in enumerate(predictions):
                        if "<sep>" in pred:
                            pred_span, pred_text = pred.split("<sep>")
                            preds.append(pred_text.strip())
                            spans.append((idx, pred_span))
                    predictions = preds

                if args.use_entire_summaries:
                    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
                elif args.use_sentence_level_summaries:
                    preds = ["\n".join([pred.strip() for pred in predictions])]
                else:
                    print("One of x must be selected")
                    exit()
                if args.filter_using_factcc:
                    if frank_p_data[entry['hash']]['FactCC'] == 1:
                        preds = ["\n".join(nltk.sent_tokenize(gen_summ.strip()))]

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

                outputs["predictions"] = {"idx": idx,
                                          "pred": " ".join([pred.strip("\n") for pred in predictions]),
                                          "ref": original_summary,
                                          "old_summ": gen_summ,
                                          "hash": entry["hash"],
                                          "model_name": entry["model_name"],
                                          "split": entry["split"]}

                if args.generate_incorrect_span:
                    outputs["predictions"]["err_span"] = str(spans)
                result_aggregator.append(outputs)

                for summary_sent in predictions:
                    summary_sent = summary_sent.strip("\n")
                    if summary_sent == "":
                        continue
                    datum = {"text": source_article,
                             "claim": summary_sent,
                             "label": "CORRECT",
                             "hash": entry["hash"],
                             "model_name": entry["model_name"],
                             "split": entry["split"]}
                    fccp.write(json.dumps(datum)+"\n")
                daep.write(str(source_article).replace("\n", " ")+"\n"+str(" ".join([pred.strip("\n") for pred in predictions])).replace("\n", " ")+"\n\n")
                daep_adj.write(json.dumps({"hash": entry["hash"],
                                "model_name": entry["model_name"],
                                "split": entry["split"]}) + "\n")
                writer.write(json.dumps(outputs)+"\n")

            metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
            aggregated_metrics = {}
            aggregated_metrics = {'cnn': {'test': {'bart': {}, 'bert_sum': {}, 'bus': {}, 'pgn': {}, 's2s': {}},
                                          'valid': {'bart': {}, 'bert_sum': {}, 'bus': {}, 'pgn': {}, 's2s': {}}},
                                  'xsum': {'test': {'BERTS2S': {}, 'TConvS2S': {}, 'PtGen': {}, 'TranS2S': {}},
                                           'valid': {'BERTS2S': {}, 'TConvS2S': {}, 'PtGen': {}, 'TranS2S': {}}}}
            for dataset in ['cnn', 'xsum']:
                for data_split in ['test', 'valid']:
                    for model_name in aggregated_metrics[dataset][data_split].keys():
                        for metric_name in metric_names:
                            aggregated_metrics[dataset][data_split][model_name][f'{split}_{metric_name}'] = []
                            if args.generate_factuality_label:
                                aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_incorrect'] = []
                                aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_correct'] = []
                                aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_num'] = []
                                if args.use_sentence_level_summaries:
                                    aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_summ_incorrect'] = []

            predictions = []
            for pred in result_aggregator:
                predictions.extend(pred["predictions"])
                data_split = pred["predictions"]['split']
                model_name = pred["predictions"]["model_name"]
                hash = pred["predictions"]['hash']
                dataset = 'cnn'
                if len(hash) == 8:
                    dataset = 'xsum'
                del pred["predictions"]
                for key, value in pred.items():
                    aggregated_metrics[dataset][data_split][model_name][key].append(value)

            for dataset in ['cnn', 'xsum']:
                for data_split in ['test', 'valid']:
                    for model_name in aggregated_metrics[dataset][data_split].keys():
                        print(aggregated_metrics[dataset][data_split][model_name])
                        for key in metric_names:
                            value = aggregated_metrics[dataset][data_split][model_name][f'{split}_{key}']
                            aggregated_metrics[dataset][data_split][model_name][f'{split}_{key}'] = sum(value)/len(value)

                            if args.generate_factuality_label:
                                aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_incorrect'] = sum(aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_incorrect'])
                                aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_correct'] = sum(aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_correct'])
                                aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_num'] = sum(aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_num'])
                                if args.use_sentence_level_summaries:
                                    aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_summ_incorrect'] = sum(aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_summ_incorrect'])/len(aggregated_metrics[dataset][data_split][model_name][f'{split}_fact_summ_incorrect'])

            print(aggregated_metrics)
            fp = open(args.output_dir+"/frank_metrics.json", "w")
            fp.write(json.dumps(aggregated_metrics))
            fp.close()

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
