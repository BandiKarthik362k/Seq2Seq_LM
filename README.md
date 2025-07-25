# Seq2Seq_LM

GPT-style Decoder only Transformer completely build from scratch. A sequence to sequence model that can take an input and generate the next probable best word sentences. Works in the same mechanism of auto-regressive text generation for the given input prompt.

Core components
  - Token embeddings
  - Positional encoding
  - Masked Multi-head self-attention
  - Feedforward networks
  - Residual connections & layer normalization

# Requirements
  - Torch
  - Optuna
  - lightining.pytorch
  - tika

# Tokenizer
- Implemented bypte pair encoding (BPE refer Vocab_Mergerules.py) tokenizer from scratch, which splits the words into characters and build the tokens for the give the set of input words.
- The given words are split into individual characters and then the most occurred consecutive characters are merged together first and this continues for the give number of iterations.
- all these merges are noted down so that for a new word the same merge rules are used again.
- added <eos>, end of sequence token to each sentence end , so that the model knows where the sentence ends and also <bos> token to recognise the sentence begin.
- bpe_encode takes a new input sequence and uses the stored merge rules, vocab ( which are created during the initial training of the tokenizer, using Vocab_Mergerules.py)

# Transformer 
- Decoder only trasnformer, which takes the tokens built by the BPE tokenizer and create the random token embedings.
- for the positions indexing, custom position embeding has been created for the model to know the places of the word sequences.
- multi layer module list is used so that the layers are dense to understand the nonlinearity of the data in higher order dimentions ( this is as part of the full feed formward network).
- As part of masked self attention : Lower triangular mask is used so that the model will not peak into the next word while learning.
- used pre-norm and post-norm along with the residual connection to improve the gradient flow and stabilize the training.
- weighted the weights to equal the emebding token weights.

# Parameter Tuning
- used optuna to get the best parameters for the training.
- tuned parameters like learning rate, weight decay, number of layers, heads, actiavtion (Relu or Gelu), feed forward netwerk internal dimention.
- used hyperband pruner on runing 25 trials and 30 epochs each, to get a view in which direction the parameters go to yield the best crossentropy loss.

# Training
# Phase 1 : Held out validation set but relavant information
- Used openly avaiable data on machine learning to train the model and also used self written notes on machine learning and deep learning.
- even though the validtion has no over lap with the training set (no leakage done)
- but the model ended up memorizing the information, because there are instances where repeated similar kind of sentences are found.
- model ended up giving almost 0.7 cross entropy loss.
- 
But this is unreliable as the model will fail in generalisation to the new type of text.

# Phase 2 : Held out validation set is with no relavence to training set at all.
- model ended up giving me a leat loss of 4.2 cross entropy loss.
- which is about perplexity of 66, that is it is getting confused among the 44 words for the next word prediction 
- model stoped with early stoppage.
# Phase 3 : Held out validation set is with no relavence to training set at all (with increased data and vocab).
- model ended up giving me a leat loss of 3.8 cross entropy loss.
- which is about perplexity of 44, that is it is getting confused among the 44 words for the next word prediction.
- model stoped with early stoppage, so there is no overfitting.

Which shows that the Transfomer is working correctly, but it's just that the model is data starving, more the less the perplexity it will be ( which is true to the working of deep leanrning and trasnformers).
considering 20000 voacbualry, that's the best the transfomer can do. on more data it will perform the best.
   
