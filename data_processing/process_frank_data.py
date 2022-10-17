import os, json
import nltk
import datasets
from data_processing.error_generator import SentenceAnnotator

sentence_annotator = SentenceAnnotator()
rouge = datasets.load_metric('rouge')
bert_score = datasets.load_metric("bertscore")

def get_sents(source_text):
    source_sents = []
    doc = sentence_annotator.spacy_nlp(source_text)
    for sent in doc.sents:
        text = sent.text
        source_sents.append(text)
    return source_sents

frank_data = json.load(open('data/benchmark_data.json'))
#json_wp = open('data/frank_benchmark_data_factcorr.json', 'w')
data = {'cnn': {'test': {}, 'valid': {}},
        'xsum': {'test': {}, 'valid': {}}}

output_dir = 'output_dir/bart_large_cnn/frank_eval/'
output_test_preds_file = os.path.join(output_dir, "frank_test_generations.json")
fcc_test_file = os.path.join(output_dir, "data-dev.jsonl")
dae_test_file = os.path.join(output_dir, "dae_input.txt")
dae_test_file_adj = os.path.join(output_dir, "dae_input_adj.txt")
result_aggregator = []
counter = 0
with open(output_test_preds_file, "w") as writer, open(fcc_test_file, "w") as fccp, open(dae_test_file, "w") as daep, open(dae_test_file_adj, 'w') as daep_adj:
    for data_point in frank_data:
        if counter % 100 == 0:
            print(counter)
        counter+=1
        #data_point = json.loads(line)
        hash = data_point['hash']
        dataset = 'cnn'
        if len(hash) == 8:
            dataset = 'xsum'
        item = {'hash': hash,
                'article': data_point['article'],
                'summary': data_point['summary'],
                'reference': data_point['reference']}

        source_article_sentences = get_sents(data_point["article"])
        generated_summary_sentences = get_sents(data_point["summary"])
        original_summary_sentences = get_sents(data_point["reference"])
        item = {"source_article_sentences": source_article_sentences,
                      "original_summary_sentences": original_summary_sentences,
                      "generated_summary_sentences": generated_summary_sentences,
                      "label": 0,
                      "error_type": "None",
                      "incorrect_sent_idx": 0,
                      "relevant_article_sent_indices": [],
                      "original_summary_sent": "None",
                      "generated_summary_sent": "None",
                      "err_span": "None",
                      "corr_span": "None",
                      "hash": hash,
                      "model_name": data_point["model_name"],
                      "split": data_point["split"]}
        #json_wp.write(json.dumps(item) + "\n")

        if data_point['model_name'] not in data[dataset][data_point['split']].keys():
            data[dataset][data_point['split']][data_point['model_name']] = []
        data[dataset][data_point['split']][data_point['model_name']].append(item)

        preds = ["\n".join(generated_summary_sentences)]
        summary = ["\n".join(original_summary_sentences)]
        outputs = {}

        # Compute rouge
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        results = rouge.compute(predictions=preds, references=summary)
        bert_metric_names = ['precision', 'recall', 'f1']
        bertscore_results = bert_score.compute(predictions=preds, references=summary, lang='en')
        for metric_name in metric_names:
            metric_val = results[metric_name].mid.fmeasure
            outputs[f'{metric_name}'] = metric_val
        for metric_name in bert_metric_names:
            metric_val = sum(bertscore_results[metric_name])/len(bertscore_results[metric_name])
            outputs[f'bertscore_{metric_name}'] = metric_val

        outputs["metadata"] = {"hash": data_point["hash"],
                                  "model_name": data_point["model_name"],
                                  "split": data_point["split"]}
        result_aggregator.append(outputs)

        for summary_sent in generated_summary_sentences:
            summary_sent = summary_sent.strip("\n")
            if summary_sent == "":
                continue
            datum = {"text": data_point["article"],
                     "claim": summary_sent,
                     "label": "INCORRECT",
                     "hash": data_point["hash"],
                     "model_name": data_point["model_name"],
                     "split": data_point["split"]}
            fccp.write(json.dumps(datum)+"\n")
        daep.write(str(data_point["article"]).replace("\n", " ")+"\n"+
                   str(" ".join([pred.strip("\n") for pred in generated_summary_sentences])).replace("\n", " ")+"\n\n")
        daep_adj.write(json.dumps({"hash": data_point["hash"],
                                   "model_name": data_point["model_name"],
                                   "split": data_point["split"]}) + "\n")
        writer.write(json.dumps(outputs)+"\n")

metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
aggregated_metrics = {'cnn': {'test': {'bart': {}, 'bert_sum': {}, 'bus': {}, 'pgn': {}, 's2s': {}},
                              'valid': {'bart': {}, 'bert_sum': {}, 'bus': {}, 'pgn': {}, 's2s': {}}},
                      'xsum': {'test': {'BERTS2S': {}, 'TConvS2S': {}, 'PtGen': {}, 'TranS2S': {}},
                               'valid': {'BERTS2S': {}, 'TConvS2S': {}, 'PtGen': {}, 'TranS2S': {}}}}
for dataset in ['cnn', 'xsum']:
    for data_split in ['test', 'valid']:
        for model_name in aggregated_metrics[dataset][data_split].keys():
            for metric_name in metric_names:
                aggregated_metrics[dataset][data_split][model_name][f'{metric_name}'] = []

for pred in result_aggregator:
    data_split = pred["metadata"]['split']
    model_name = pred["metadata"]["model_name"]
    hash = pred["metadata"]['hash']
    dataset = 'cnn'
    if len(hash) == 8:
        dataset = 'xsum'
    del pred["metadata"]
    for key, value in pred.items():
        aggregated_metrics[dataset][data_split][model_name][key].append(value)

for dataset in ['cnn', 'xsum']:
    for data_split in ['test', 'valid']:
        for model_name in aggregated_metrics[dataset][data_split].keys():
            for key in metric_names:
                value = aggregated_metrics[dataset][data_split][model_name][f'{key}']
                aggregated_metrics[dataset][data_split][model_name][f'{key}'] = sum(value)/len(value)

print(aggregated_metrics)
fp = open(output_dir+"/frank_metrics.json", "w")
fp.write(json.dumps(aggregated_metrics))
fp.close()


for dataset in ['cnn', 'xsum']:
    for split in ['test', 'valid']:
        for model_name in data[dataset][split]:
            data[dataset][split][model_name] = len(data[dataset][split][model_name])

print(data)
