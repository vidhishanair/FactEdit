import logging
import os
import re
import sys
import copy
import json

import nltk  # Here to have a nice missing dependency error message early on
import spacy
import numpy as np
from datasets import load_dataset, load_metric
from levenstein import Distance

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from fact_trainer.train_utils import ModelArguments, DataTrainingArguments
from fact_trainer.train_utils import get_last_checkpoint, summarization_name_mapping
from fact_trainer.data_collator import DataCollatorForSeq2Seq
from fact_trainer.train import safely_get_last_checkpoint, setup_logging, get_datasets

logger = logging.getLogger(__name__)

spacy_nlp = spacy.load('en_core_web_lg')

def main():
    #output_dir = "/remote/bones/user/vbalacha/summary_fact_corrector/output_dir/cnndmv3_bart_base_seq2seq_sentcorrect_relevpass_addfullsummctxt_INFILLErr_ErP0.2_data1.0train1.0val1.0test_bs12_ep1_gacc2/"
    #output_dir = "/remote/bones/user/vbalacha/summary_fact_corrector/output_dir/xsum/xsum_bart_base_seq2seq_sentcorrect_relevpass_addfullsummctxt_INFILLErr_ErP0.2_data0.5train0.5val0.5test_bs12_ep1_gacc2_gpu2/predict2/"
    output_dir = "/remote/bones/user/vbalacha/summary_fact_corrector/output_dir/cnndmv3_bart_base_seq2seq_sentcorrect_relevpass_addfullsummctxt_INFILLErr_ErP0.2_data1.0train1.0val1.0test_bs12_ep1_gacc2/"
    test_examples = []
    for line in open(os.path.join(output_dir, "data-dev.jsonl")):
        datum = json.loads(line)
        if len(test_examples)>0 and datum["text"] == test_examples[-1]["text"]:
            continue
        test_examples.append(datum)

    fact_corr_outputs = []
    for line in open(os.path.join(output_dir, "test_generations.json")):
        item = json.loads(line)['predictions']
        fact_corr_outputs.append((item['idx'], item["pred"], item['ref'], item['old_summ']))

    bart_outputs = []
    for line in open("data/bart_test_sent.json"):
    #for line in open("data/bart_xsum_test_sent_wspan_wid.json"):
        item = json.loads(line)
        # bart_outputs.append(" ".join(item["generated_summary_sentences"]))
        bart_outputs.append(item["generated_summary_sentences"])

    wp = open(os.path.join(output_dir, "error_comp.tsv"), "w")
    count = 0
    for example, bout, fact_corr_outs in zip(test_examples, bart_outputs, fact_corr_outputs):
        idx = fact_corr_outs[0]
        fact_corr_out = fact_corr_outs[1]
        ref = fact_corr_outs[2]
        bout = bart_outputs[idx]

        # if bout != fact_corr_out:
        #     dist = nltk.edit_distance(bout, fact_corr_out)
        #     count += 1
        #     if count > 100:
        #         break
        #     # example = json.loads(example)
        #     source_art = example["text"].replace("\t"," ").replace("\n", " ")
        #     # wp.write(example[1].replace("\t"," ").replace("\n", " ")+"\t"+example[0].replace("\t"," ").replace("\n", " ")+"\t"+bout.replace("\t"," ").replace("\n", " ")+"\t"+esw_out.replace("\t"," ").replace("\n", " ")+"\n")
        #     wp.write(source_art+"\t"+ref.replace("\t"," ").replace("\n", " ")+"\t"+bout.replace("\t"," ").replace("\n", " ")+"\t"+fact_corr_out.replace("\t"," ").replace("\n", " ")+"\t"+str(dist)+"\t"+str(bout!=fact_corr_out)+"\n")

        doc = spacy_nlp(fact_corr_out)
        fact_corr_out_sents = []
        for sent in doc.sents:
            text = sent.text
            fact_corr_out_sents.append(text)

        for refs, outs in zip(bout, fact_corr_out_sents):
            if refs != outs:
                dist = nltk.edit_distance(refs, outs)
                refop, hypop, errop, wer = Distance().Visualize(refs.split(), outs.split(), sys.stdout)
                count += 1
                if count > 100:
                    break
                source_art = example["text"].replace("\t"," ").replace("\n", " ")
                wp.write(source_art+"\t"+
                         ref.replace("\t"," ").replace("\n", " ")+"\t"+
                         " ".join(bout).replace("\t"," ").replace("\n", " ")+"\t"+
                         fact_corr_out.replace("\t"," ").replace("\n", " ")+"\t"+
                         str(dist)+"\t"+
                         str(refop)+"\t"+
                         str(hypop)+"\t"+
                         str(errop)+"\t"+
                         str(wer)+"\t"+
                         "\n")


if __name__ == "__main__":
    main()

