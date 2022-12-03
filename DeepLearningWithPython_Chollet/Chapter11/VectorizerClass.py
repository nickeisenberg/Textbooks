import string

class Vectorizer:


    def standardize(self, text):
        text = text.lower()
        return ''.join([char for char in text
                        if char not in string.punctuation])

    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()

    def make_vocabulary(self, dataset):
        self.vocabulary = {'': 0, '[UNK]': 1}
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
            self.inverse_vocabulary = dict(
                    (v, k) for k, v in self.vocabulary.items())
                
    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return ' '.join(
                self.inverse_vocabulary.get(i, '[UNK}') for i in int_sequence)

        
vectorizer = Vectorizer()

dataset = ['I write, erase, rewrite',
           'Erase again, and then',
           'A poppy blooms',
           ]

vectorizer.make_vocabulary(dataset)
print(vectorizer.inverse_vocabulary)
print(vectorizer.vocabulary)
test_sentacne = 'I write, rewrite, an still write again'
encoded_sentance = vectorizer.encode(test_sentacne)
print(encoded_sentance)
decoded_sentance = vectorizer.decode(encoded_sentance)
print(decoded_sentance)


