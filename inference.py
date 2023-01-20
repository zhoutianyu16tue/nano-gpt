import time
from acl_model import Model
import numpy as np
from scipy.special import softmax
from numpy.random import multinomial

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('/path/to/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

device_id = 0
model_path = "/path/to/nanoGPT.om"
acl_model = Model(device_id, model_path)
vocab_size = 65
block_size = 256
batch_size = 1
text_len_to_generate = 1000

idx = [0]
for i in range(text_len_to_generate):
    # crop idx to the last block_size tokens
    if len(idx) < block_size:
        idx_tmp = idx + [0] * (block_size-len(idx))
    else:
        idx_tmp = idx[-block_size:]
    idx_cond = np.array(idx_tmp, dtype=np.int32).reshape(1, block_size)
    logits = acl_model.infer(idx_cond).reshape((batch_size, block_size, vocab_size))
    
    ii = i if i < block_size else -1
    logits = logits[:, ii, :] # becomes (B, C)

    # apply softmax to get probabilities
    probs = softmax(logits.astype(np.double), axis=-1) # (B, C)
    tmp = multinomial(1, probs[0], 1)
    idx_next = np.argmax(tmp, axis=1)
    # append sampled index to the running sequence
    idx.append(idx_next[0])

print(decode(idx))
