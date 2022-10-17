import os, json, time
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset, load_metric

from data_processing.error_generator import SentenceAnnotator

sentence_annotator = SentenceAnnotator()

def get_sents(source_text):
    source_sents = []
    doc = sentence_annotator.spacy_nlp(source_text)
    for sent in doc.sents:
        text = sent.text
        source_sents.append(text)
    return source_sents

spacy_nlp = spacy.load('en_core_web_lg')
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", force_bos_token_to_be_generated=True).to('cuda')
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

dataset = load_dataset("cnn_dailymail", "3.0.0")
rouge = load_metric('rouge')
bert_score = load_metric("bertscore")
rouge_metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
bertscore_metric_names = ['precision', 'recall', 'f1']
rouge_results = {}
bertscore_results = {}
for metric_name in rouge_metric_names:
    rouge_results[metric_name] = []
for metric_name in bertscore_metric_names:
    bertscore_results[metric_name] = []

# fcc_test_file = os.path.join('output_dir/bart_large_cnn', "data-dev.jsonl")
# dae_test_file = os.path.join('output_dir/bart_large_cnn', "dae_input.txt")
fcc_test_file = os.path.join('output_dir/bart_large_cnn/factcc_corr', "data-dev.jsonl")
dae_test_file = os.path.join('output_dir/bart_large_cnn/factcc_corr', "dae_input.txt")

#json_wp = open("data/bart_test_sent.json", "w")
start_data = time.process_time()
with open(fcc_test_file, "w") as fccp, open(dae_test_file, "w") as daep:
    for idx, data in enumerate(dataset["test"]):
        if idx % 1000 == 0:
            print("Processed "+str(idx)+" examples in: "+str(time.process_time() - start_data))

        source_text = data["article"]#.strip("\n")
        summary = data["highlights"]#.strip("\n")
        source_article_sentences = get_sents(source_text)
        inputs = tokenizer([source_text], max_length=1024, return_tensors='pt').to('cuda')

        # Generate Summary
        #summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=128, early_stopping=True)
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=140, min_length=55, length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)
        generated_summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    
        summary_sents = get_sents(summary.strip())
        error_summary_sents = get_sents(generated_summary[0].strip())
        data_point = {"source_article_sentences": source_article_sentences,
                "original_summary_sentences": summary_sents,
                "generated_summary_sentences": error_summary_sents,
                "label": 0,
                "error_type": "None",
                "incorrect_sent_idx": 0,
                "relevant_article_sent_indices": [],
                "original_summary_sent": "None",
                "generated_summary_sent": "None"}
        #json_wp.write(json.dumps(data_point) + "\n")

        preds = ["\n".join(error_summary_sents)]
        summary = ["\n".join(summary_sents)]
        rouge_result = rouge.compute(predictions=preds, references=summary, use_stemmer=True)
        bertscore_result = bert_score.compute(predictions=preds, references=summary, lang='en')
        for metric_name in rouge_metric_names:
            metric_val = rouge_result[metric_name].mid.fmeasure
            rouge_results[metric_name].append(metric_val)
        for metric_name in bertscore_metric_names:
            metric_val = sum(bertscore_result[metric_name])/len(bertscore_result[metric_name])
            bertscore_results[metric_name].append(metric_val)

        #summary_doc = spacy_nlp(pred)
        #for summary_sent in summary_sents:
        for sid, summary_sent in enumerate(error_summary_sents):
            summary_sent = summary_sent.strip("\n")
            #if summary_sent == "":
            #    continue
            datum = {"text": source_text,
                     "claim": summary_sent,
                     "label": "CORRECT",
                     "sent_id": sid,
                     "idx": idx,
                     "cnn_id": data["id"]}
            fccp.write(json.dumps(datum)+"\n")
        daep.write(str(source_text)+"\n"+str(generated_summary[0])+"\n\n")

for metric_name in rouge_metric_names:
    rouge_results[metric_name] = sum(rouge_results[metric_name])/len(rouge_results[metric_name])
for metric_name in bertscore_metric_names:
    bertscore_results[metric_name] = sum(bertscore_results[metric_name])/len(bertscore_results[metric_name])

res_fp = open('output_dir/bart_large_cnn/factcc_corr/results.txt', 'w')
res_fp.write(json.dumps(rouge_results)+"\n")
res_fp.write(json.dumps(bertscore_results)+"\n")
print(rouge_results)
print(bertscore_results)
