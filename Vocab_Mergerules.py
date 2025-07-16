from collections import Counter

words = Counter()
#req = ['a','i']
with open("data\corpus\Final_Train_Text.txt", 'r') as f:
    for i in f:
        for word in i.split():
            current = tuple(word.lower()) + ('</w>',)
            words[current] = words[current] + 1

total_raw_corpus = len(words)
print(total_raw_corpus)

def adj_pairs(words):
    adj_pairs = Counter()
    for k,v in words.items():
        for a,b in zip(k,k[1:]):
            if b=='</w>':
                continue
            adj_pairs[(a,b)] = adj_pairs[(a,b)] + v
    if not adj_pairs:
        return None,None
    return adj_pairs.most_common(1)[0][0][0], adj_pairs.most_common(1)[0][0][1]

def modify_vocab(words):
    first, second = adj_pairs(words)
    if first is None:
        return None,None,None
    club = first + second
    updated_words = Counter()
    for k,v in words.items():
        current_word = []
        i=0
        while i < len(k):
            if i+1< len(k) and k[i] == first and k[i+1] == second:
                current_word.append(club)
                i=i+2
            else:
                current_word.append(k[i])
                i=i+1
        updated_words[tuple(current_word)] =  updated_words[tuple(current_word)] + v
    return updated_words, first, second

merges = []
for _ in range(20000):
    new_words,first,second = modify_vocab(words)
    if first is None:
        break
    words = new_words
    merges.append((first, second))

with open("data\corpus\MergeRules.txt",'w') as f:
    for i,j in merges:
        f.write(f"{i} {j}\n")

token_set = {'unk', '<pad>', '</w>', '<bos>', '<eos>'}
for seq in words.keys():
    for piece in seq:
        if piece == '</w>':
            continue
        token_set.add(piece)

ordered_tokens = ['unk','<pad>','</w>','<bos>','<eos>']
for a,b in merges:
    club = a+b
    if club not in ordered_tokens:
        ordered_tokens.append(club)
for tok in sorted(token_set):
    if tok not in ordered_tokens:
        ordered_tokens.append(tok)

with open("data\corpus\Vocab.txt", "w", encoding="utf-8") as f:
    for tok in ordered_tokens:
        f.write(tok + "\n")
