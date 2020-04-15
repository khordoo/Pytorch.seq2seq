import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.01
HIDDEN_SIZE = 100
EMBEDDINGS_DIMS = 50
TEACHER_ENFORCE_PROB = 0.5
conversation_pair = [
    ['Hi how are you?', 'I am good ,thank you.'],
    ['How was your day?', 'It was a fantastic day'],
    ['Good morning!', 'Good morning to you too'],
    ['How everything is going', 'Things are going great'],
]


class EncoderRNN(nn.Module):
    """Simple RNN module with word embeddings"""

    def __init__(self, input_size, hidden_size, embeddings_dims):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.lstm = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, hidden_states):
        x = self.emb(x)
        out, hidden_states = self.lstm(x, hidden_states)
        return out, hidden_states


class DecoderRNN(nn.Module):
    """Simple decoder class with embedding and a linear layer to project the out put of
       the LSTM to a vocabulary size dimension.  n_hidden_size -> number_vocabulary
    """

    def __init__(self, input_size, hidden_size, embeddings_dims, vocab_size):
        super(DecoderRNN, self).__init__()
        self.emb = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.lstm = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_states):
        x = self.emb(x)
        out, hidden_states = self.lstm(x, hidden_states)
        out = self.linear(out)
        return out, hidden_states


class Tokenizer:
    """Converts Text into its numerical representation"""

    def __init__(self):
        self.START_TOKEN = '<sos>'
        self.word2index = {self.START_TOKEN: 0}
        self.index2word = {0: self.START_TOKEN}
        self.words_count = 1

    def fit_on_text(self, text_array):
        """Creates a numerical index value for every unique word"""
        for sentence in text_array:
            self._add_sentence(sentence)

    def _add_sentence(self, sentence):
        """Creates indexes for unique word in the sentences and
        adds them to the dictionary"""
        for word in sentence.strip().lower().split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.words_count
                self.index2word[self.words_count] = word
                self.words_count += 1

    def texts_to_index(self, sentences):
        """Convert words in sentences to their numerical index values"""
        sentences_index = []
        for sentence in sentences:
            sentences_index.append([
                self.word2index[word]
                for word in sentence.strip().lower().split(' ')
            ])
        return sentences_index

    def indexes_to_text(self, word_numbers):
        """Converts an array of numbers to a text sentence"""
        return ' '.join([self.index2word[idx] for idx in word_numbers])

    @property
    def start_token_index(self):
        return self.word2index[self.START_TOKEN]


class TrainingSession:
    """A container class that runs the training job"""

    def __init__(self, encoder, decoder, tokenizer, device, learning_rate, teacher_enforce_prob):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.teacher_enforce_prob = teacher_enforce_prob
        self.start_token_index = self.tokenizer.start_token_index

    def train(self, sources, targets, num_epoc=100):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)

        for epoc in range(num_epoc):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            for source, target in zip(sources, targets):
                loss = 0
                # LSTM has two internal states (h,c)
                encoder_hidden = (torch.zeros(1, 1, self.encoder.hidden_size).to(self.device),
                                  torch.zeros(1, 1, self.encoder.hidden_size).to(self.device))

                for word_idx in source:
                    encoder_input = torch.LongTensor([[word_idx]]).to(self.device)
                    encoder_out, encoder_hidden = self.encoder(encoder_input, encoder_hidden)

                decoder_hidden = encoder_hidden
                decoder_input = torch.LongTensor([[self.start_token_index]]).to(self.device)
                predicted_indexes = []
                for target_idx in target:
                    decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    predicted_target = decoder_out.argmax(dim=2)
                    actual_target = torch.LongTensor([[target_idx]]).to(self.device)
                    if np.random.random() < self.teacher_enforce_prob:
                        decoder_input = actual_target
                    else:
                        decoder_input = predicted_target

                    loss += F.cross_entropy(decoder_out.squeeze(0), actual_target.flatten())
                    predicted_indexes.append(decoder_out.argmax(dim=2).item())

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                if epoc % 5 == 0:
                    print(f'epoc:{epoc}, loss: {loss.item()}')
                    print(self.tokenizer.indexes_to_text(source))
                    print(self.tokenizer.indexes_to_text(predicted_indexes))


sources, targets = zip(*conversation_pair)
tokenizer = Tokenizer()
tokenizer.fit_on_text(sources + targets)
sources = tokenizer.texts_to_index(sources)
targets = tokenizer.texts_to_index(targets)
encoder = EncoderRNN(input_size=tokenizer.words_count, hidden_size=HIDDEN_SIZE, embeddings_dims=EMBEDDINGS_DIMS).to(
    DEVICE)
decoder = DecoderRNN(input_size=tokenizer.words_count, hidden_size=HIDDEN_SIZE, embeddings_dims=EMBEDDINGS_DIMS,
                     vocab_size=tokenizer.words_count).to(DEVICE)
trainer = TrainingSession(encoder=encoder, decoder=decoder, tokenizer=tokenizer, learning_rate=LEARNING_RATE,
                          teacher_enforce_prob=TEACHER_ENFORCE_PROB,
                          device=DEVICE)
trainer.train(sources, targets, num_epoc=15)
