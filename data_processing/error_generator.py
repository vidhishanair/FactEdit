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
from data_processing.perplexity_scorer import score_ppl


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
    parser.add_argument('--entswap_err', action='store_true', default=False, help='Generate entity swap errors')
    parser.add_argument('--incorsubj_err', action='store_true', default=False, help='Generate incorrect subject errors')
    parser.add_argument('--ooasubj_err', action='store_true', default=False, help='Generate ooa subject errors')
    parser.add_argument('--incorobj_err', action='store_true', default=False, help='Generate incorrect object errors')
    parser.add_argument('--ooaobj_err', action='store_true', default=False, help='Generate ooa object errors')
    parser.add_argument('--incorrel_err', action='store_true', default=False, help='Generate incorrect relation errors')
    parser.add_argument('--ooarel_err', action='store_true', default=False, help='Generate ooa relation errors')
    parser.add_argument('--gen_infill_data', action='store_true', default=False, help='Generate data for infilling model')

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
    print('Datadir: '+str(dirname))
    print("overwriting cache: "+str(write_cache))
    if write_cache:
        if not os.path.exists(args.preprocess_cache+"/"+split):
            os.makedirs(args.preprocess_cache+"/"+split)
    
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
            if write_cache:
                cache_wp = open(args.preprocess_cache+"/"+split+"/"+str(idx)+".json", "w")
            if args.write_json_datapoint:
                json_wp = open(dirname+"/"+str(idx)+".json", "w")
            if args.write_tsvs:
                wp = open(dirname+"/"+str(idx)+".tsv", 'w')
                write_tsv_line("Source Article", "Summary", "Summary Original Sent", "Original PPL",
                               "Summary Error Sent", "Error Sent PPL", "Error Type", "Summary Original Sent Triple",
                               "Random Source Triple", "Source Triples", wp)
        idx += 1
        if args.dataset == 'cnn':
            source_text = data["article"].strip("\n")
            summary = data["highlights"].strip("\n")
        else:
            source_text = data["document"].strip("\n")
            summary = data["summary"].strip("\n")

        if write_cache:
            source_triples, source_sents = get_source_triples(source_text, sentence_annotator)
            summary_sentences = []
            if args.dataset == 'cnn':
                summary_doc = sentence_annotator.spacy_nlp(summary)
                summary_doc_sents = [sent for sent in summary_doc.sents]
            else:
                summary_doc_sents = [summary]
            for summary_sent in summary_doc_sents: #summary_doc.sents:
                #summary_sent = summary_sent.text.strip("\n")
                summary_sent = summary_sent.strip("\n")
                summary_sent_ppl = score_ppl(summary_sent)
                summary_sentences.append({"sent_text": summary_sent,
                                          "sent_ppl": summary_sent_ppl})
    
                oie_frames = sentence_annotator.get_oie_frames(summary_sent)
                summary_sentences[-1]["oie_frames"] = oie_frames
            
            if len(source_sents) > 0:
                rouge_scores = compute_batched_sentence_rouge(source_sents, [x["sent_text"] for x in summary_sentences], metric)
                for sent_idx, sent in enumerate(summary_sentences):
                    relevant_article_sent_indices = []
                    relevant_article_sents = []
                    sent_summ_art_rouge = rouge_scores[sent_idx]["rouge_scores"]
                    top_k_rouge_sents = sorted(sent_summ_art_rouge, key=lambda x: x["rougeLsum"], reverse=True)[0:3]
                    for rouge_sents in top_k_rouge_sents:
                        article_idx = rouge_sents["article_sent_idx"]
                        relevant_article_sent_indices.append(article_idx)
                        relevant_sents = source_sents[article_idx-1:article_idx+2]
                        relevant_article_sents.append(str(relevant_sents))
                    summary_sentences[sent_idx]["relevant_article_sent_indices"] = relevant_article_sent_indices
            else:
                for sent_idx, sent in enumerate(summary_sentences):
                    summary_sentences[sent_idx]["relevant_article_sent_indices"] = []
    
            datum = {"article": source_text, "highlights": summary, "source_sents": source_sents,
                     "source_triples": source_triples, "summary_sentences": summary_sentences}
            cache_wp.write(json.dumps(datum)+"\n")
            continue
        else:
            source_triples = data["source_triples"]
            source_sents = data["source_sents"]
            summary_sentences = data["summary_sentences"]
    
        if args.debug:
            print("Source Triples: "+str(time.process_time() - start))
        if len(source_triples) == 0:
            continue
        dataset_triples.extend(source_triples)

        for summary_sent_data in summary_sentences:
            summary_sent = summary_sent_data["sent_text"]
            summary_sent_ppl = summary_sent_data["sent_ppl"]
            oie_frames = summary_sent_data["oie_frames"]
            relevant_article_sent_indices = summary_sent_data["relevant_article_sent_indices"]
    
            for verb in oie_frames['verbs']:
                oie_spans = sentence_annotator.get_oie_spans(oie_frames, verb, summary_sent)
                if len(oie_spans["verb_spans"]) == 0:
                    continue

    
                # Swap subj and object
                if args.entswap_err and len(oie_spans["arg0_spans"]) != 0 and len(oie_spans["arg1_spans"]) != 0:
                    error_type = "Swap Entities"
                    start = time.process_time()
                    if oie_spans["arg0_spans"][0]["start_pos"] < oie_spans["arg1_spans"][0]["start_pos"]:
                        replaced_sent = summary_sent[0:oie_spans["arg0_spans"][0]["start_pos"]] \
                                        + oie_spans["arg1_spans"][0]["text"] \
                                        + summary_sent[
                                          oie_spans["arg0_spans"][0]["end_pos"]:oie_spans["arg1_spans"][0]["start_pos"]] \
                                        + oie_spans["arg0_spans"][0]["text"] \
                                        + summary_sent[oie_spans["arg1_spans"][0]["end_pos"]:]
                    else:
                        replaced_sent = summary_sent[0:oie_spans["arg1_spans"][0]["start_pos"]] \
                                        + oie_spans["arg0_spans"][0]["text"] \
                                        + summary_sent[
                                          oie_spans["arg1_spans"][0]["end_pos"]:oie_spans["arg0_spans"][0]["start_pos"]] \
                                        + oie_spans["arg1_spans"][0]["text"] \
                                        + summary_sent[oie_spans["arg0_spans"][0]["end_pos"]:]
                    if args.debug:
                        print("Entswap Err Replaced sent: "+str(time.process_time() - start))
                    start = time.process_time()
                    write_to_files(source_text, source_sents, summary, summary_sentences, summary_sent,
                                   summary_sent_ppl, replaced_sent, relevant_article_sent_indices, error_type, oie_spans, "",
                                   source_triples, wp, json_wp, args)
                    if args.debug:
                        print("Entswap Err: "+str(time.process_time() - start))
    
                if len(oie_spans["arg0_spans"]) != 0:
                    # Replace subj with other subjs from source
                    if args.incorsubj_err:
                        start = time.process_time()
                        error_type = "Incorrect Subject"
                        replacement_arg0_spans = None
                        random_choice_count = 0
                        while replacement_arg0_spans is None:
                            random_triple = random.choice(source_triples)
                            if len(random_triple["arg0_spans"]) == 0 or \
                                    (len(random_triple["arg0_spans"][0]['tokenized_text']) >
                                     2*len(oie_spans["arg0_spans"][0]['tokenized_text'])):
                                if random_choice_count == 10:
                                    break
                                random_choice_count += 1
                                continue
                            replacement_arg0_spans = random_triple["arg0_spans"][0]
                            replaced_sent = summary_sent[0:oie_spans["arg0_spans"][0]["start_pos"]] \
                                            + replacement_arg0_spans["text"] \
                                            + summary_sent[oie_spans["arg0_spans"][0]["end_pos"]:]
                            if args.debug:
                                print("Incor Subj Err Replaced sent: "+str(time.process_time() - start))
                            start = time.process_time()
                            write_to_files(source_text, source_sents, summary, summary_sentences, summary_sent,
                                           summary_sent_ppl, replaced_sent, relevant_article_sent_indices,
                                           error_type, oie_spans, random_triple, source_triples, wp, json_wp, args)
                            if args.debug:
                                print("Incor Subj Err: "+str(time.process_time() - start))
    
                    if args.ooasubj_err and idx > 100:
                        start = time.process_time()
                        # Replace subj with random subjs from dataset
                        error_type = "Out of Art Subj"
                        replacement_arg0_spans = None
                        random_choice_count = 0
                        while replacement_arg0_spans is None:
                            random_triple = random.choice(dataset_triples)
                            if len(random_triple["arg0_spans"]) == 0 or \
                                    (len(random_triple["arg0_spans"][0]['tokenized_text']) >
                                     2*len(oie_spans["arg0_spans"][0]['tokenized_text'])):
                                if random_choice_count == 10:
                                    break
                                random_choice_count += 1
                                continue
                            replacement_arg0_spans = random_triple["arg0_spans"][0]
                            replaced_sent = summary_sent[0:oie_spans["arg0_spans"][0]["start_pos"]] \
                                            + replacement_arg0_spans["text"] \
                                            + summary_sent[oie_spans["arg0_spans"][0]["end_pos"]:]
                            if args.debug:
                                print("OOA Subj Replaced sent: "+str(time.process_time() - start))
                            start = time.process_time()
                            write_to_files(source_text, source_sents, summary, summary_sentences, summary_sent,
                                           summary_sent_ppl, replaced_sent, relevant_article_sent_indices,
                                           error_type, oie_spans, random_triple, source_triples, wp, json_wp, args)
                            if args.debug:
                                print("OOA Subj Err: "+str(time.process_time() - start))
    
                if len(oie_spans["arg1_spans"]) != 0:
                    # Replace obj with other objs from source
                    if args.incorobj_err:
                        start = time.process_time()
                        error_type = "Incorrect Obj"
                        replacement_arg1_spans = None
                        random_choice_count = 0
                        while replacement_arg1_spans is None:
                            random_triple = random.choice(source_triples)
                            if len(random_triple["arg1_spans"]) == 0 or \
                                    (len(random_triple["arg1_spans"][0]['tokenized_text']) >
                                     2*len(oie_spans["arg1_spans"][0]['tokenized_text'])):
                                if random_choice_count == 10:
                                    break
                                random_choice_count += 1
                                continue
                            replacement_arg1_spans = random_triple["arg1_spans"][0]
                            replaced_sent = summary_sent[0:oie_spans["arg1_spans"][0]["start_pos"]] \
                                            + replacement_arg1_spans["text"] \
                                            + summary_sent[oie_spans["arg1_spans"][0]["end_pos"]:]
                            if args.debug:
                                print("Incor Obj Replaced sent: "+str(time.process_time() - start))
                            start = time.process_time()
                            write_to_files(source_text, source_sents, summary, summary_sentences, summary_sent,
                                           summary_sent_ppl, replaced_sent, relevant_article_sent_indices,
                                           error_type, oie_spans, random_triple, source_triples, wp, json_wp, args)
                            if args.debug:
                                print("Incor Obj Err: "+str(time.process_time() - start))
    
                    if args.ooaobj_err and idx > 100:
                        # Replace obj with random objs from dataset
                        start = time.process_time()
                        error_type = "Out of Art Obj"
                        replacement_arg1_spans = None
                        random_choice_count = 0
                        while replacement_arg1_spans is None:
                            random_triple = random.choice(dataset_triples)
                            if len(random_triple["arg1_spans"]) == 0 or \
                                    (len(random_triple["arg1_spans"][0]['tokenized_text']) >
                                     2*len(oie_spans["arg1_spans"][0]['tokenized_text'])):
                                if random_choice_count == 10:
                                    break
                                random_choice_count += 1
                                continue
                            replacement_arg1_spans = random_triple["arg1_spans"][0]
                            replaced_sent = summary_sent[0:oie_spans["arg1_spans"][0]["start_pos"]] \
                                            + replacement_arg1_spans["text"] \
                                            + summary_sent[oie_spans["arg1_spans"][0]["end_pos"]:]
                            if args.debug:
                                print("OOA Obj Replaced sent: "+str(time.process_time() - start))
                            start = time.process_time()
                            write_to_files(source_text, source_sents, summary, summary_sentences, summary_sent,
                                           summary_sent_ppl, replaced_sent, relevant_article_sent_indices,
                                           error_type, oie_spans, random_triple, source_triples, wp, json_wp, args)
                            if args.debug:
                                print("OOA Obj Err: "+str(time.process_time() - start))
    
                # Replace pred with other preds from source
                if args.incorrel_err:
                    start = time.process_time()
                    error_type = "Incorrect Rel"
                    replacement_verb_spans = None
                    random_choice_count = 0
                    while replacement_verb_spans is None:
                        random_triple = random.choice(source_triples)
                        if len(random_triple["verb_spans"]) == 0 :
                            if random_choice_count == 10:
                                break
                            random_choice_count += 1
                            continue
                        replacement_verb_spans = random_triple["verb_spans"][0]
                        replaced_sent = summary_sent[0:oie_spans["verb_spans"][0]["start_pos"]] \
                                        + replacement_verb_spans["text"] \
                                        + summary_sent[oie_spans["verb_spans"][0]["end_pos"]:]
                        if args.debug:
                            print("Incor Rel Replaced sent: "+str(time.process_time() - start))
                        start = time.process_time()
                        write_to_files(source_text, source_sents, summary, summary_sentences, summary_sent,
                                       summary_sent_ppl, replaced_sent, relevant_article_sent_indices,
                                       error_type, oie_spans, random_triple, source_triples, wp, json_wp, args)
                        if args.debug:
                            print("Incor Rel Err: "+str(time.process_time() - start))
    
                if args.ooarel_err and idx > 100:
                    # Replace obj with random objs from dataset
                    start = time.process_time()
                    error_type = "Out of Art Rel"
                    replacement_verb_spans = None
                    random_choice_count = 0
                    while replacement_verb_spans is None:
                        random_triple = random.choice(dataset_triples)
                        if len(random_triple["verb_spans"]) == 0:
                            if random_choice_count == 10:
                                break
                            random_choice_count += 1
                            continue
                        replacement_verb_spans = random_triple["verb_spans"][0]
                        replaced_sent = summary_sent[0:oie_spans["verb_spans"][0]["start_pos"]] \
                                        + replacement_verb_spans["text"] \
                                        + summary_sent[oie_spans["verb_spans"][0]["end_pos"]:]
                        if args.debug:
                            print("OOA Rel Replaced sent: "+str(time.process_time() - start))
                        start = time.process_time()
                        write_to_files(source_text, source_sents, summary, summary_sentences, summary_sent,
                                       summary_sent_ppl, replaced_sent, relevant_article_sent_indices,
                                       error_type, oie_spans, random_triple, source_triples, wp, json_wp, args)
                        if args.debug:
                            print("OOA Rel Err: "+str(time.process_time() - start))
