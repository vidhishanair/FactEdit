import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
model_id = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


def score_ppl(sentence):
    sentence = sentence.lower()
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    loss = model(tensor_input, labels=tensor_input)
    return math.exp(loss[0])


if __name__ == '__main__':
    a=['there is a book on the desk',
       'there is a plane on the desk',
       'there is a book in the desk']
    print([score_ppl(i) for i in a])
