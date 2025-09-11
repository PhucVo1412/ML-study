corpus = ["Tôi thích môn Toán","Tôi thích AI", "Tôi thích âm nhạc"]

vocab = set()

def bag_of_words(corpus, sentence):
    for sentence1 in corpus:
        for word in sentence1.split():
            vocab.add(word)
    print('vocalbary:',vocab)

    vocal_map = dict((word,0) for word in vocab)

    for word in sentence.split():
        if word in vocab:
            vocal_map[word] += 1
    return list(vocal_map.values())

print(bag_of_words(corpus, "Tôi thích AI thích Toán"))

