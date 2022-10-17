import os, time
import json
import argparse
from pathlib import Path
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import spacy
import random
from datasets import load_metric
# from data_processing.perplexity_scorer import score_ppl


class SentenceAnnotator():
    def __init__(self, oie=False, srl=False, chunker=False):
        # load spacy
        self.spacy_nlp = spacy.load('en_core_web_lg')

        if oie:
            # load OIE
            self.oie_predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")

        if srl:
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


sentence_annotator = SentenceAnnotator(oie=True)


def get_source_triples(source_text):
    source_triples = []
    source_sents = []
    doc = sentence_annotator.spacy_nlp(source_text)
    for sent in doc.sents:
        text = sent.text
        source_sents.append(text)
    return source_triples, source_sents


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


def get_data(write_cache, preprocess_cache):
    print("Loading Dataset")
    print(write_cache)
    fp = open(preprocess_cache_file)
    for line in fp:
        yield json.loads(line)


def compute_batched_sentence_rouge(article_sentences, summary_sentences, rouge_obj):
    sentence_level_rouge = []
    preds = []
    refs = []
    for idx1, summary_sent in enumerate(summary_sentences):
        # preds.extend([summary_sent]*len(article_sentences))
        # refs.extend(article_sentences)
        preds.extend(article_sentences)
        refs.extend([summary_sent["sent_text"]]*len(article_sentences))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--write_tsvs', action='store_true', default=False, help='Write tsv files for visualization')
    parser.add_argument('--write_json_datapoint', action='store_true', default=False, help='Write json data files')
    parser.add_argument('--debug', action='store_true', default=False, help='Print debugging info')
    parser.add_argument('--ppl_threshold', type=float, default=80.00, help='PPL Threshold')
    parser.add_argument('--summary_error_prob', type=float, default=0.7, help='Probability of error in summary')
    parser.add_argument('--swap_error_prob', type=float, default=0.9, help='Probability of error being swap error')
    parser.add_argument('--start', type=int, default=0, help='Start from idx')
    parser.add_argument('--end', type=int, default=5000, help='End at idx')
    parser.add_argument('--overwrite_cache', action='store_true', default=False, help='Flag to overwrite cache')
    parser.add_argument('--data_path', type=str, default="/remote/bones/user/vbalacha/summary_fact_corrector/data/processed_wsourcesents/", help='Data path')
    parser.add_argument('--preprocess_cache', type=str, default="data/processed/", help='Cache path')
    parser.add_argument('--split', type=str, default="train", help='Data Split')
    parser.add_argument('--entswap_err', action='store_true', default=False, help='Generate entity swap errors')
    parser.add_argument('--incorsubj_err', action='store_true', default=False, help='Generate incorrect subject errors')
    parser.add_argument('--ooasubj_err', action='store_true', default=False, help='Generate ooa subject errors')
    parser.add_argument('--incorobj_err', action='store_true', default=False, help='Generate incorrect object errors')
    parser.add_argument('--ooaobj_err', action='store_true', default=False, help='Generate ooa object errors')
    parser.add_argument('--incorrel_err', action='store_true', default=False, help='Generate incorrect relation errors')
    parser.add_argument('--ooarel_err', action='store_true', default=False, help='Generate ooa relation errors')

    args = parser.parse_args()


    split = args.split
    preprocess_cache_file = Path(args.data_path+split+".json")
    write_cache = True
    wp = None
    json_wp = None
    idx = 0
    cache_wp = open(args.preprocess_cache+"/"+split+"/"+str(idx)+".json", "w")
    metric_name = "rouge"
    metric = load_metric(metric_name)

    dataset_triples = []
    for data in get_data(write_cache, preprocess_cache_file):
        if idx < args.start:
            idx += 1
            continue
        if idx >= args.end:
            break
        if idx % 1000 == 0:
            print(idx)
            if write_cache:
               cache_wp = open(args.preprocess_cache+"/"+split+"/"+str(idx)+".json", "w")
        idx += 1
        source_text = data["article"].strip("\n")
        source_triples, source_sents = get_source_triples(source_text)
        data["source_sents"] = source_sents
        source_article_sents = data["source_sents"]
        summary = data["highlights"].strip("\n")
        summary_sentences = data["summary_sentences"]

        start = time.process_time()
        if write_cache:
            rouge_scores = compute_batched_sentence_rouge(source_article_sents, summary_sentences, metric)
            for sent_idx, sent in enumerate(summary_sentences):
                relevant_article_sent_indices = []
                relevant_article_sents = []
                sent_summ_art_rouge = rouge_scores[sent_idx]["rouge_scores"]
                top_k_rouge_sents = sorted(sent_summ_art_rouge, key=lambda x: x["rougeLsum"], reverse=True)[0:3]
                for rouge_sents in top_k_rouge_sents:
                    article_idx = rouge_sents["article_sent_idx"]
                    relevant_article_sent_indices.append(article_idx)
                    relevant_sents = source_article_sents[article_idx-1:article_idx+2]
                    relevant_article_sents.append(str(relevant_sents))
                data["summary_sentences"][sent_idx]["relevant_article_sent_indices"] = relevant_article_sent_indices
                # tsv_file.write(str(source_article_sents)+"\t"+str(summary_sentences)+"\t"+
                #                str(sent["sent_text"])+"\t"+" |||| ".join(relevant_article_sents)+"\t"+
                #                str(data["summary_sentences"])+"\n")
            cache_wp.write(json.dumps(data)+"\n")
            continue
        else:
            source_triples = data["source_triples"]
            summary_sentences = data["summary_sentences"]
