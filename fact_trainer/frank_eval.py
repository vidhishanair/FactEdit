import json
import argparse
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
#parser.add_argument("--filter_using_factcc", type=bool, required=True)
parser.add_argument('--filter_using_factcc', dest='filter_using_factcc', default=False, action='store_true')

factcc_aggregated_metrics = {'cnn': {'test': {'bart': {}, 'bert_sum': {}, 'bus': {}, 'pgn': {}, 's2s': {}},
                              'valid': {'bart': {}, 'bert_sum': {}, 'bus': {}, 'pgn': {}, 's2s': {}}},
                      'xsum': {'test': {'BERTS2S': {}, 'TConvS2S': {}, 'PtGen': {}, 'TranS2S': {}},
                               'valid': {'BERTS2S': {}, 'TConvS2S': {}, 'PtGen': {}, 'TranS2S': {}}}}
args = parser.parse_args()
output_dir = args.output_dir


#baseline_factuality_metrics_outputs = open('data/baseline_factuality_metrics_outputs.json')
frank_data = json.load(open('data/baseline_factuality_metrics_outputs.json'))
frank_p_data = {'cnn': {'test': {}, 'valid': {}},
        'xsum': {'test': {}, 'valid': {}}}
for data_point in frank_data:
    frank_p_data[data_point['hash']] = data_point

frank_data = json.load(open('data/frank_human_annotations.json'))
frank_human_data = {'cnn': {'test': {}, 'valid': {}},
        'xsum': {'test': {}, 'valid': {}}}
for data_point in frank_data:
    frank_human_data[data_point['hash']] = data_point

fp = open(output_dir+"data-dev.jsonl")
fpp = open(output_dir+"factcc_eval_predictions.txt")

for fp_line in fp:
    fpp_line = json.loads(fpp.readline().strip("\n"))
    facc_data = json.loads(fp_line)
    facc_pred = int(fpp_line['pred'])
    data_split = facc_data['split']
    model_name = facc_data["model_name"]
    hash = facc_data['hash']
    dataset = 'cnn'
    if len(hash) == 8:
        dataset = 'xsum'
    if not args.filter_using_factcc:
        # if hash not in factcc_aggregated_metrics[dataset][data_split][model_name].keys():
        #     factcc_aggregated_metrics[dataset][data_split][model_name][hash] = 0
        # if facc_pred == 1:
        #     factcc_aggregated_metrics[dataset][data_split][model_name][hash] = 1
        
        if hash not in factcc_aggregated_metrics[dataset][data_split][model_name].keys():
            factcc_aggregated_metrics[dataset][data_split][model_name][hash] = 1
        if facc_pred == 1:
            factcc_aggregated_metrics[dataset][data_split][model_name][hash] = 0
    else:
        if frank_p_data[hash]['FactCC'] == 1:
            factcc_aggregated_metrics[dataset][data_split][model_name][hash] = 1
        else:
            if hash not in factcc_aggregated_metrics[dataset][data_split][model_name].keys():
                factcc_aggregated_metrics[dataset][data_split][model_name][hash] = 1
            if facc_pred == 1:
                factcc_aggregated_metrics[dataset][data_split][model_name][hash] = 0

for dataset in ['cnn', 'xsum']:
    for split in ['test', 'valid']:
        for model_name in factcc_aggregated_metrics[dataset][split]:
            labels = [v for k, v in factcc_aggregated_metrics[dataset][split][model_name].items()]
            factcc_aggregated_metrics[dataset][split][model_name] = f1_score(y_true=[1]*len(labels), y_pred=labels, average="micro")

print(factcc_aggregated_metrics)

dae_aggregated_metrics = {'cnn': {'test': {'bart': {}, 'bert_sum': {}, 'bus': {}, 'pgn': {}, 's2s': {}},
                                     'valid': {'bart': {}, 'bert_sum': {}, 'bus': {}, 'pgn': {}, 's2s': {}}},
                             'xsum': {'test': {'BERTS2S': {}, 'TConvS2S': {}, 'PtGen': {}, 'TranS2S': {}},
                                      'valid': {'BERTS2S': {}, 'TConvS2S': {}, 'PtGen': {}, 'TranS2S': {}}}}

dae_adj = [json.loads(line) for line in open(output_dir+'dae_input_adj.txt').readlines()]
input_data = [line.strip() for line in open(output_dir+"dae_preds.txt").readlines()]
counter = 0
for idx in range(0, len(input_data), 4):
    article_text = input_data[idx]
    summary = input_data[idx + 1]
    pred = int(input_data[idx + 2])
    metadata = dae_adj[counter]
    counter += 1
    data_split = metadata['split']
    model_name = metadata["model_name"]
    hash = metadata['hash']
    dataset = 'cnn'
    if len(hash) == 8:
        dataset = 'xsum'
    if not args.filter_using_factcc:
        # if hash not in dae_aggregated_metrics[dataset][data_split][model_name].keys():
        #     dae_aggregated_metrics[dataset][data_split][model_name][hash] = 0
        # if pred == 1:
        #     dae_aggregated_metrics[dataset][data_split][model_name][hash] = 1
        if hash not in dae_aggregated_metrics[dataset][data_split][model_name].keys():
            dae_aggregated_metrics[dataset][data_split][model_name][hash] = 1
        if pred == 0:
            dae_aggregated_metrics[dataset][data_split][model_name][hash] = 0
    else:
        if frank_p_data[hash]['FactCC'] == 1:
            dae_aggregated_metrics[dataset][data_split][model_name][hash] = 1
        else:
            if hash not in dae_aggregated_metrics[dataset][data_split][model_name].keys():
                dae_aggregated_metrics[dataset][data_split][model_name][hash] = 1
            if pred == 0:
                dae_aggregated_metrics[dataset][data_split][model_name][hash] = 0


for dataset in ['cnn', 'xsum']:
    for split in ['test', 'valid']:
        for model_name in dae_aggregated_metrics[dataset][split]:
            fact_preds = [v for k, v in dae_aggregated_metrics[dataset][split][model_name].items()]
            dae_aggregated_metrics[dataset][split][model_name] = float(sum(fact_preds))/len(fact_preds)

print(dae_aggregated_metrics)
