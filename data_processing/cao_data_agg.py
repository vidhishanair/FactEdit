import json

from data_processing.error_generator import SentenceAnnotator

sentence_annotator = SentenceAnnotator()

def get_sents(source_text):
    source_sents = []
    doc = sentence_annotator.spacy_nlp(source_text)
    for sent in doc.sents:
        text = sent.text
        source_sents.append(text)
    return source_sents

# files = ['train-clean.jsonl', 'train-pronoun.jsonl', 'train-dateswp.jsonl', 'train-numswp.jsonl', 'train-entswp.jsonl']
# error_type = ['None', 'Pronoun', 'Date Swap', 'Num Swap', 'Ent Swap']
# label = [0, 1, 1, 1, 1]
# data_count = [201644, 20204,  16858, 13408, 35113]
# #wp = open('/remote/bones/user/vbalacha/Factual-Error-Correction/cnn-dailymail/train_data/train.jsonl', 'w')
# wp = open('/remote/bones/user/vbalacha/Factual-Error-Correction/xsum/train_data/train.jsonl', 'w')

# #files = ['val-clean.jsonl', 'val-pronoun.jsonl', 'val-dateswp.jsonl', 'val-numswp.jsonl', 'val-entswp.jsonl']
# files = ['validation-clean.jsonl', 'validation-pronoun.jsonl', 'validation-dateswp.jsonl', 'validation-numswp.jsonl', 'validation-entswp.jsonl']
# error_type = ['None', 'Pronoun', 'Date Swap', 'Num Swap', 'Ent Swap']
# label = [0, 1, 1, 1, 1]
# data_count = [5710, 1445,  1445, 1445, 1445]
# #wp = open('/remote/bones/user/vbalacha/Factual-Error-Correction/cnn-dailymail/train_data/val.jsonl', 'w')
# wp = open('/remote/bones/user/vbalacha/Factual-Error-Correction/xsum/train_data/validation.json', 'w')

files = ['test-clean.jsonl', 'test-pronoun.jsonl', 'test-dateswp.jsonl', 'test-numswp.jsonl', 'test-entswp.jsonl']
error_type = ['None', 'Pronoun', 'Date Swap', 'Num Swap', 'Ent Swap']
label = [0, 1, 1, 1, 1]
data_count = [5710, 1445,  1445, 1445, 1445]
# wp = open('/remote/bones/user/vbalacha/Factual-Error-Correction/cnn-dailymail/train_data/test.jsonl', 'w')
wp = open('/remote/bones/user/vbalacha/Factual-Error-Correction/xsum/train_data/test.json', 'w')


for f, etype, label, dcount in zip(files, error_type, label, data_count):
    count = 0
    #for line in open('/remote/bones/user/vbalacha/Factual-Error-Correction/cnn-dailymail/'+f):
    for line in open('/remote/bones/user/vbalacha/Factual-Error-Correction/xsum/'+f):
        count += 1
        if count > dcount:
            continue
        if count % 100 == 0:
            print("Processed "+str(count))
        item = json.loads(line)
        source_article_sentences = get_sents(item['text'])
        summary_sents = get_sents(item['summary'])
        error_summary_sents = get_sents(item['claim'])
        data_point = {"source_article_sentences": source_article_sentences,
                      "original_summary_sentences": summary_sents,
                      "generated_summary_sentences": error_summary_sents,
                      "label": label,
                      "error_type": etype,
                      "incorrect_sent_idx": -1,
                      "relevant_article_sent_indices": [],
                      "original_summary_sent": "None",
                      "generated_summary_sent": "None",
                      "err_span": "None",
                      "corr_span": "None"}
        wp.write(json.dumps(data_point)+"\n")
