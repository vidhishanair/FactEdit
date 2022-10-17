import os
import time
import json
import argparse
import multiprocessing
import torch
from pathlib import Path
#from allennlp.predictors.predictor import Predictor
#import allennlp_models.tagging
import spacy
import random
from datasets import load_dataset, load_metric
#from data_processing.perplexity_scorer import score_ppl


class SentenceAnnotator():
    def __init__(self, oie=False, srl=False, chunker=False):
        # load spacy
        self.spacy_nlp = spacy.load('en_core_web_lg')
        #from allennlp.predictors.predictor import Predictor

        if oie:
            from allennlp.predictors.predictor import Predictor
            # load OIE
            cuda_device = 0 if torch.cuda.is_available() else -1
            self.oie_predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", cuda_device=cuda_device)

        if srl:
            from allennlp.predictors.predictor import Predictor
            # load SRL
            self.srl_predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

        if chunker:
            # load tagger
            from flair.models import SequenceTagger
            self.chunker = SequenceTagger.load("flair/chunk-english-fast")

    def get_oie_frames(self, text):
        oie_results = self.oie_predictor.predict(sentence=text)
        return oie_results

    def get_oie_spans(self, oie_results, verb, text):
        oie_spans = {"verb_spans": [],
                     "arg0_spans": [],
                     "arg1_spans": [],
                     "arg2_spans": [],
                     "argm_spans": []}
        start_offset = 0
        end_offset = 0
        offset = 0
        prev_tag = None
        current_span = []
        for tok, tag in zip(oie_results['words'], verb['tags']):
            if text[offset] == ' ':
                offset += 1
            if tag == 'O' or tag.split('-')[0] == 'B':
                if prev_tag is not None and prev_tag != 'O':
                    if prev_tag.split('-')[1] == 'V':
                        oie_spans["verb_spans"].append({"tokenized_text": " ".join(current_span),
                                                        "text": text[start_offset:end_offset],
                                                        "start_pos": start_offset, "end_pos": end_offset})
                    if prev_tag.split('-')[1] == 'ARG0':
                        oie_spans["arg0_spans"].append({"tokenized_text": " ".join(current_span),
                                                        "text": text[start_offset:end_offset],
                                                        "start_pos": start_offset, "end_pos": end_offset})
                    if prev_tag.split('-')[1] == 'ARG1':
                        oie_spans["arg1_spans"].append({"tokenized_text": " ".join(current_span),
                                                        "text": text[start_offset:end_offset],
                                                        "start_pos": start_offset, "end_pos": end_offset})
                    if prev_tag.split('-')[1] == 'ARG2':
                        oie_spans["arg2_spans"].append({"tokenized_text": " ".join(current_span),
                                                        "text": text[start_offset:end_offset],
                                                        "start_pos": start_offset, "end_pos": end_offset})
                    if prev_tag.split('-')[1] == 'ARGM':
                        oie_spans["argm_spans"].append({"tokenized_text": " ".join(current_span),
                                                        "text": text[start_offset:end_offset],
                                                        "start_pos": start_offset, "end_pos": end_offset})
                current_span = []
                start_offset = offset
                end_offset = offset
            prev_tag = tag
            offset += len(tok)
            end_offset = offset
            if tag == 'O':
                continue
            current_span.append(tok)
        return oie_spans

    def get_chunker_spans(self, text):
        # make example sentence
        from flair.data import Sentence
        from flair.tokenization import SpacyTokenizer
        sentence = Sentence(text, use_tokenizer=SpacyTokenizer(model="en_core_web_lg"))
        # sentence = Sentence(text)
        # predict NER tags

        self.chunker.predict(sentence)
        spans = []
        tok_to_span_map = {}
        for entity in sentence.get_spans('np'):
            span_text = text[entity.start_pos:entity.end_pos]
            span_type = entity.annotation_layers['np'][0]._value  # check 0 and np assumption
            spans.append((span_text, span_type))
            for tok in entity:
                tok_text = tok.text
                tok_to_span_map[tok_text] = (span_text, span_type)
        return spans, tok_to_span_map


def get_sents(source_text, sentence_annotator):
    source_sents = []
    doc = sentence_annotator.spacy_nlp(source_text)
    for sent in doc.sents:
        text = sent.text
        source_sents.append(text)
    return source_sents


def get_source_triples(source_text, sentence_annotator):
    source_triples = []
    source_sents = get_sents(source_text, sentence_annotator)
    for text in source_sents:
        oie_frames = sentence_annotator.get_oie_frames(text)
        for verb in oie_frames['verbs']:
            oie_spans = sentence_annotator.get_oie_spans(oie_frames, verb, text)
            if len(oie_spans["arg0_spans"]) == 0:
                continue
            source_triples.append(oie_spans)
    return source_triples, source_sents


def write_tsv_line(source_text, summary, summary_sent, summary_sent_ppl, replaced_sent, replaced_sent_ppl,
                   error_type, oie_spans, source_replacement_triple, source_triples, wp):
    wp.write(source_text.replace("\n", " ") + "\t"
             + summary.replace("\n", " ") + "\t"
             + summary_sent.replace("\n", " ") + "\t"
             + str(summary_sent_ppl) + "\t"
             + replaced_sent.replace("\n", " ") + "\t"
             + str(replaced_sent_ppl) + "\t"
             + error_type + "\t"
             + str(oie_spans) + "\t"
             + str(source_replacement_triple) + "\t"
             + str(len(source_triples)) + "\t"
             + str(source_triples).replace("\n", " ") + "\n")


def write_json_datapoint(source_article, source_article_sentences, summary, summary_sents,
                         chosen_summary_sent, replaced_summary_sent, relevant_article_sent_indices,
                         label, error_type, json_wp):
    error_summary_sents = []
    chosen_sent_idx = None
    summary_sents = [x["sent_text"] for x in summary_sents]
    for idx, sent in enumerate(summary_sents):
        if sent == chosen_summary_sent and chosen_sent_idx is None:
            chosen_sent_idx = idx
            error_summary_sents.append(replaced_summary_sent)
        else:
            error_summary_sents.append(sent)

    data = {"source_article_sentences": source_article_sentences,
            "original_summary_sentences": summary_sents,
            "generated_summary_sentences": error_summary_sents,
            "incorrect_sent_idx": chosen_sent_idx,
            "relevant_article_sent_indices": relevant_article_sent_indices,
            # "original_summary": summary.replace("\n", " "),
            # "generated_summary": error_summary.replace("\n", " "),
            "original_summary_sent": chosen_summary_sent,
            "generated_summary_sent": replaced_summary_sent,
            "label": label,
            "error_type": error_type}
    json_wp.write(json.dumps(data) + "\n")


def write_to_files(source_article, source_article_sentences, summary, summary_sents, chosen_summary_sent,
                   summary_sent_ppl, replaced_summary_sent, relevant_article_sent_indices, error_type,
                   oie_spans, source_replacement_triple, source_triples, wp, json_wp, args):
    start = time.process_time()
    replaced_sent_ppl = None
    if args.ppl_threshold != -1:
        replaced_sent_ppl = score_ppl(replaced_summary_sent)
    if args.debug:
        print("PPL: "+str(time.process_time() - start))
    if args.ppl_threshold == -1 or replaced_sent_ppl < args.ppl_threshold:
        if args.write_json_datapoint:
            error_prob = random.random()
            # swap_error_prob = random.random()
            if error_prob > args.summary_error_prob: # and swap_error_prob > 0.9:
                write_json_datapoint(source_article, source_article_sentences, summary, summary_sents,
                                     chosen_summary_sent, replaced_summary_sent, relevant_article_sent_indices,
                                     0, error_type, json_wp)
            else:
                write_json_datapoint(source_article, source_article_sentences, summary, summary_sents,
                                     chosen_summary_sent, chosen_summary_sent, relevant_article_sent_indices,
                                     1, "None", json_wp)
        if args.write_tsvs:
            write_tsv_line(source_article, summary, chosen_summary_sent, summary_sent_ppl, replaced_summary_sent,
                           replaced_sent_ppl, error_type, oie_spans, source_replacement_triple, source_triples, wp)


def compute_batched_sentence_rouge(article_sentences, summary_sentences, rouge_obj):
    sentence_level_rouge = []
    preds = []
    refs = []
    for idx1, summary_sent in enumerate(summary_sentences):
        # preds.extend([summary_sent]*len(article_sentences))
        # refs.extend(article_sentences)
        preds.extend(article_sentences)
        refs.extend([summary_sent]*len(article_sentences))
    rouge_score = rouge_obj.compute(predictions=preds, references=refs, use_agregator=False)
    for idx1, abs_sent in enumerate(summary_sentences):
        filtered_rouge = {"summ_sent_idx": idx1, "rouge_scores":[]}
        for idx, val in enumerate(article_sentences):
            datum = {"article_sent_idx": idx}
            for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                datum[metric] = rouge_score[metric][(idx1*len(article_sentences))+idx].fmeasure
            filtered_rouge["rouge_scores"].append(datum)
        sentence_level_rouge.append(filtered_rouge)
    return sentence_level_rouge


def get_errtype_dirname(args):
    error_set = "Err"
    if args.entswap_err:
        error_set += "_ESW"
    if args.incorsubj_err:
        error_set += "_ISU"
    if args.incorobj_err:
        error_set += "_IOB"
    if args.ooasubj_err:
        error_set += "_OSU"
    if args.ooaobj_err:
        error_set += "_OOB"
    if args.incorrel_err:
        error_set += "_IRL"
    if args.ooarel_err:
        error_set += "_ORL"
    error_set += "_ErP"+str(args.summary_error_prob)
    return error_set


def get_data(dataset_name, write_cache, preprocess_cache_file):
    print("Loading Dataset")
    if not write_cache:
        fp = open(preprocess_cache_file)
        for line in fp:
            yield json.loads(line)
    else:
        if dataset_name == 'cnn':
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            dataset = dataset[split]
            for idx, data in enumerate(dataset):
                yield data
        else:
            dataset = load_dataset("xsum")
            dataset = dataset[split]
            for idx, data in enumerate(dataset):
                yield data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--write_tsvs', action='store_true', default=False, help='Write tsv files for visualization')
    parser.add_argument('--write_json_datapoint', action='store_true', default=False, help='Write json data files')
    parser.add_argument('--debug', action='store_true', default=False, help='Print debugging info')
    parser.add_argument('--ppl_threshold', type=float, default=80.00, help='PPL Threshold')
    parser.add_argument('--summary_error_prob', type=float, default=0.7, help='Probability of error in summary')
    parser.add_argument('--swap_error_prob', type=float, default=0.9, help='Probability of error being swap error')
    parser.add_argument('--overwrite_cache', action='store_true', default=False, help='Flag to overwrite cache')
    parser.add_argument('--preprocess_cache', type=str, default="data/processed/", help='Cache path')
    parser.add_argument('--dir_save_prefix', type=str, default="test", help='Cache path')
    parser.add_argument('--dataset', type=str, default="cnn", help='Dataset name')
    parser.add_argument('--split', type=str, default="train", help='Data Split')
    parser.add_argument('--size', type=int, default=300000, help='Start from idx')
    parser.add_argument('--interval', type=int, default=5000, help='Start from idx')
    parser.add_argument('--start', type=int, default=0, help='Start from idx')
    parser.add_argument('--end', type=int, default=5000, help='End at idx')
    parser.add_argument('--gen_source_infill_data', action='store_true', default=False, help='Generate data for infilling model')
    parser.add_argument('--gen_summ_infill_data', action='store_true', default=False, help='Generate data for infilling model')
    parser.add_argument('--entswap_err', action='store_true', default=False, help='Generate entity swap errors')
    parser.add_argument('--incorsubj_err', action='store_true', default=False, help='Generate incorrect subject errors')
    parser.add_argument('--ooasubj_err', action='store_true', default=False, help='Generate ooa subject errors')
    parser.add_argument('--incorobj_err', action='store_true', default=False, help='Generate incorrect object errors')
    parser.add_argument('--ooaobj_err', action='store_true', default=False, help='Generate ooa object errors')
    parser.add_argument('--incorrel_err', action='store_true', default=False, help='Generate incorrect relation errors')
    parser.add_argument('--ooarel_err', action='store_true', default=False, help='Generate ooa relation errors')
     

    args = parser.parse_args()

    if not args.write_tsvs and not args.write_json_datapoint and not args.overwrite_cache:
        print("Atleast one of write_tsvs or write_json_datapoint or overwrite_cache need to be provided")
        exit()

    split = args.split
    preprocess_cache_file = Path(args.preprocess_cache+split+".json")
    write_cache = True
    if preprocess_cache_file.exists() and not args.overwrite_cache:
        write_cache = False

    metric_name = "rouge"
    metric = load_metric(metric_name)

    if write_cache:
        oie = True
    else:
        oie = False
    sentence_annotator = SentenceAnnotator(oie=oie)
    wp = None
    json_wp = None
    cache_wp = None
    idx = 0
    error_set = get_errtype_dirname(args)
    dirname = "data/"+args.dir_save_prefix+"_"+error_set+"/"+split
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not os.path.exists("data/"+args.dir_save_prefix+"_"+error_set+"/masked_references/"):
        os.makedirs("data/"+args.dir_save_prefix+"_"+error_set+"/masked_references/")
    if not os.path.exists("data/"+args.dir_save_prefix+"_"+error_set+"/masked_references/"+split):
        os.makedirs("data/"+args.dir_save_prefix+"_"+error_set+"/masked_references/"+split)
    print('Datadir: '+str(dirname))

    dataset_triples = []
    start_data = time.process_time()
    for data in get_data(args.dataset, write_cache, preprocess_cache_file):
        start = time.process_time()
        if idx < args.start:
            idx += 1
            continue
        if idx >= args.end:
            break
        if idx % 1000 == 0:
            print("Processed "+str(idx)+" examples in: "+str(time.process_time() - start_data))
            if args.write_json_datapoint:
                json_wp = open(dirname+"/"+str(idx)+".json", "w")
                summ_json_wp = open("data/"+args.dir_save_prefix+"_"+error_set+"/masked_references/"+split+"/"+str(idx)+".json", "w")

        idx += 1
        if args.dataset == 'cnn':
            source_text = data["article"].strip("\n")
            summary = data["highlights"].strip("\n")
        else:
            source_text = data["document"].strip("\n")
            summary = data["summary"].strip("\n")

        source_triples = data["source_triples"]
        source_sents = data["source_sents"]
        summary_sentences = data["summary_sentences"]

        if args.debug:
            print("Source Triples: "+str(time.process_time() - start))
        if len(source_triples) == 0:
            continue
        dataset_triples.extend(source_triples)

        if args.gen_source_infill_data:
            filtered_source_sents = [x for x in source_sents if len(x.split(" ")) >= 5]
            select_idxs = []
            for vid, vspan in enumerate(source_triples[0:5]):
                for arg0_span in source_triples[vid]["arg0_spans"][0:10]:
                    if len(arg0_span["tokenized_text"].split(" ")) <= 5:
                        for idx1, source_sent in enumerate(filtered_source_sents[0:6]):
                            if " "+arg0_span["text"]+" " in source_sent:
                                select_idxs.append(idx1)
                                masked_sent = source_sent.replace(" "+arg0_span["tokenized_text"]+" ", " <mask> ")
                                if "<mask>" not in masked_sent:
                                    continue
                                target = arg0_span["tokenized_text"]
                                source = " ".join([x for i, x in enumerate(source_sents) if i != idx1]) #+ " <sep> " + masked_sent
                                datum = {"source": source, "target": target, "masked_sent": masked_sent}
                                json_wp.write(json.dumps(datum)+"\n")
                for arg1_span in source_triples[vid]["arg1_spans"][0:10]:
                    if len(arg1_span["tokenized_text"].split(" ")) <= 5:
                        for idx1, source_sent in enumerate(filtered_source_sents[0:6]):
                            if " "+arg1_span["text"]+" " in source_sent:
                                select_idxs.append(idx1)
                                masked_sent = source_sent.replace(" "+arg1_span["tokenized_text"]+" ", " <mask> ")
                                if "<mask>" not in masked_sent:
                                    continue
                                target = arg1_span["tokenized_text"]
                                source = " ".join([x for i, x in enumerate(source_sents) if i != idx1]) #+ " <sep> " + masked_sent
                                datum = {"source": source, "target": target,  "masked_sent": masked_sent}
                                json_wp.write(json.dumps(datum)+"\n")
                for verb_span in source_triples[vid]["verb_spans"][0:10]:
                    if len(verb_span["tokenized_text"].split(" ")) <= 5:
                        for idx1, source_sent in enumerate(filtered_source_sents[0:6]):
                            if " "+verb_span["text"]+" " in source_sent:
                                select_idxs.append(idx1)
                                masked_sent = source_sent.replace(" "+verb_span["tokenized_text"]+" ", " <mask> ")
                                if "<mask>" not in masked_sent:
                                    continue
                                target = verb_span["tokenized_text"]
                                source = " ".join([x for i, x in enumerate(source_sents) if i != idx1]) #+ " <sep> " + masked_sent
                                datum = {"source": source, "target": target,  "masked_sent": masked_sent}
                                json_wp.write(json.dumps(datum)+"\n")

        if args.gen_summ_infill_data:
            for summary_sent_data in summary_sentences:
                summary_sent = summary_sent_data["sent_text"]
                summary_sent_ppl = summary_sent_data["sent_ppl"]
                oie_frames = summary_sent_data["oie_frames"]
                relevant_article_sent_indices = summary_sent_data["relevant_article_sent_indices"]

                for verb in oie_frames['verbs']:
                    oie_spans = sentence_annotator.get_oie_spans(oie_frames, verb, summary_sent)
                    spans = []
                    if len(oie_spans["arg0_spans"]) != 0:
                        spans.append(oie_spans["arg0_spans"][0])
                    if len(oie_spans["arg1_spans"]) != 0:
                        spans.append(oie_spans["arg1_spans"][0])
                    if len(oie_spans["verb_spans"]) != 0:
                        spans.append(oie_spans["verb_spans"][0])
                    for span in spans:
                        masked_sent = summary_sent.replace(" "+span["tokenized_text"]+" ", " [blank] ")
                        if "[blank]" not in masked_sent:
                            continue
                        #masked_sent = summary_sent[0:span["start_pos"]] \
                        #              + "[blank]" \
                        #              + summary_sent[span["end_pos"]:]
                        target = span["tokenized_text"]
                        source = " ".join(source_sents)
                        datum = {"source_article_sentences": source_sents,
                                 "target": target,
                                 "masked_sent": masked_sent,
                                 "original_sent": summary_sent,
                                 "original_sent_ppl": summary_sent_ppl,
                                 "original_summary_sentences": [x["sent_text"] for x in summary_sentences],
                                 "relevant_article_sent_indices": relevant_article_sent_indices}
                        summ_json_wp.write(json.dumps(datum)+"\n")

                    # data = {"source_article_sentences": source_article_sentences,
                    #         "original_summary_sentences": summary_sents,
                    #         "generated_summary_sentences": error_summary_sents,
                    #         "incorrect_sent_idx": chosen_sent_idx,
                    #         "relevant_article_sent_indices": relevant_article_sent_indices,
                    #         # "original_summary": summary.replace("\n", " "),
                    #         # "generated_summary": error_summary.replace("\n", " "),
                    #         "original_summary_sent": chosen_summary_sent,
                    #         "generated_summary_sent": replaced_summary_sent,
                    #         "label": label,
                    #         "error_type": error_type}
