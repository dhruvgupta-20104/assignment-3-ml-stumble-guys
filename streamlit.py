import streamlit as st
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import numpy as np
import torch
from torch import nn

st.title("Assignment 3")

class NextChar(nn.Module):
  def __init__(self, context, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(context * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    return x
  
itos = {0: '\n', 1: ' ', 2: '!', 3: "'", 4: ',', 5: '-', 6: '.', 7: ':', 8: ';', 9: '?', 10: '[', 11: ']', 12: 'a', 13: 'b', 14: 'c', 15: 'd', 16: 'e', 17: 'f', 18: 'g', 19: 'h', 20: 'i', 21: 'j', 22: 'k', 23: 'l', 24: 'm', 25: 'n', 26: 'o', 27: 'p', 28: 'q', 29: 'r', 30: 's', 31: 't', 32: 'u', 33: 'v', 34: 'w', 35: 'x', 36: 'y', 37: 'z'}
stoi = {'\n': 0, ' ': 1, '!': 2, "'": 3, ',': 4, '-': 5, '.': 6, ':': 7, ';': 8, '?': 9, '[': 10, ']': 11, 'a': 12, 'b': 13, 'c': 14, 'd': 15, 'e': 16, 'f': 17, 'g': 18, 'h': 19, 'i': 20, 'j': 21, 'k': 22, 'l': 23, 'm': 24, 'n': 25, 'o': 26, 'p': 27, 'q': 28, 'r': 29, 's': 30, 't': 31, 'u': 32, 'v': 33, 'w': 34, 'x': 35, 'y': 36, 'z': 37}

def plot_emb(emb, itos, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    emb_extract = []
    for i in range(len(itos)):
        emb_extract.append(emb.weight[i].detach().cpu().numpy())
    emb_extract = np.array(emb_extract)
    if(emb.weight[0].shape[0] > 2):
        tsne = TSNE(n_components=2, random_state=0)
        emb_extract = tsne.fit_transform(emb_extract)
    for i in range(len(itos)):
        x, y = emb_extract[i]
        ax.scatter(x, y, color='k')
        ax.text(x + 0.05, y + 0.05, itos[i])
    return ax

def get_k_chars(input, k, model, c):
    result = input
    context = input[-c:]
    for count in range(k):
        x = [[stoi[char] for char in context]]
        x = torch.tensor(x)
        y_pred = model(x)
        y = itos[torch.distributions.categorical.Categorical(logits=y_pred).sample().item()]
        result+=y
        context = context[1:] + y
    return result

def main():
    text_input = st.text_input('Enter initial text :', '')
    k = st.slider('Select a value for number of characters to be predicted:', min_value=0, max_value=50, value=8)
    block_size = st.slider('Select a value for Block Size:', min_value=0, max_value=10, value=5)
    embedding_size = st.slider('Select a value for Embedding Size:', min_value=2, max_value=6, value=4)
    values_list = [5, 10, 15, 20, 25]
    neurons = st.slider('Select a value for number of neurons:', 0, 4, 0, format_func=lambda i: values_list[i])
    # [5, 10, 15, 20, 25] 

    model = NextChar(block_size, len(stoi), embedding_size, neurons)

    model.load_state_dict(torch.load('trained_models/{}_{}_{}'.format(block_size, embedding_size, neurons)))
    result = get_k_chars(text_input, k, model, block_size)

    if st.button('Submit'):
        st.write('The predicted next k characters are as follows: \n', result)
        # st.pyplot(plot_emb(model.emb, itos))

if __name__ == '__main__':
    main()

