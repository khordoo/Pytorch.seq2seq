import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.01
LSTM_HIDDEN_SIZE = 100
EMBEDDINGS_DIMS = 50
TEACHER_FORCING_PROB = 0.5
MAX_SEQUENCE_LENGTH = 10

conversation_pair = [
    ['Hi how are you?', 'I am good ,thank you.'],
    ['How was your day?', 'It was a fantastic day'],
    ['Good morning!', 'Good morning to you too'],
    ['How everything is going', 'Things are going great'],
]


class EncoderLSTM(nn.Module):
    """A simple decoder with word embeddings"""

    def __init__(self, input_size, hidden_size, embeddings_dims):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(num_embeddings=input_size, embedding_dim=embeddings_dims)
        self.lstm = nn.LSTM(input_size=embeddings_dims, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, hidden_states):
        x = self.emb(x)
        out, hidden_states = self.lstm(x, hidden_states)
        return out, hidden_states


class DecoderLSTM(nn.Module):
    """A simple decoder with embedding, and a linear layer to project the output of
       the layer LSTM to a vocabulary size dimension:  hidden_size -> vocab_size
    """

    def __init__(self, input_size, hidden_size, embeddings_dims, vocab_size):
        super(DecoderLSTM, self).__init__()
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

    def __init__(self, max_sequence_length):
        self.START_TOKEN = '<sos>'
        self.PADDING_TOKEN = '<pad>'
        self.END_TOKEN = '<eos>'
        self.word2index = {self.PADDING_TOKEN: 0, self.START_TOKEN: 1, self.END_TOKEN: 2}
        self.index2word = {0: self.PADDING_TOKEN, 1: self.START_TOKEN, 2: self.END_TOKEN}
        self.words_count = len(self.word2index)
        self.max_length = max_sequence_length

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
        end_token_index = self.word2index[self.END_TOKEN]
        for sentence in sentences:
            sentence_index = []
            for word in sentence.strip().lower().split(' '):
                sentence_index.append(self.word2index[word])
            sentence_index.append(end_token_index)
            sentence_index = self._pad(sentence_index)
            sentence_index = self._clip(sentence_index)
            sentences_index.append(sentence_index)
        return sentences_index

    def _clip(self, sequence):
        return sequence[:self.max_length]

    def _pad(self, sequence):
        pad_index = self.word2index[self.PADDING_TOKEN]
        while len(sequence) < self.max_length:
            sequence.append(pad_index)
        return sequence

    def indexes_to_text(self, word_numbers):
        """Converts an array of numbers to a text string"""
        ignore_index = [self.word2index[self.PADDING_TOKEN],
                        self.word2index[self.END_TOKEN]
                        ]
        return ' '.join([self.index2word[idx] for idx in word_numbers if idx not in ignore_index])

    @property
    def start_token_index(self):
        return self.word2index[self.START_TOKEN]


class TrainingSession:
    """A container class that runs the training job"""

    def __init__(self, encoder, decoder, tokenizer, device, learning_rate, teacher_forcing_prob):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.teacher_forcing_prob = teacher_forcing_prob
        self.start_token_index = self.tokenizer.start_token_index
        self.pad_token_index = self.tokenizer.word2index[self.tokenizer.PADDING_TOKEN]
        self.end_token_index = self.tokenizer.word2index[self.tokenizer.END_TOKEN]

    def train(self, sources, targets, epochs=20):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        batch_size = len(sources)
        for epoch in range(epochs):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_hidden = (torch.zeros(1, batch_size, self.encoder.hidden_size).to(self.device),
                              torch.zeros(1, batch_size, self.encoder.hidden_size).to(self.device))

            encoder_input = torch.LongTensor(sources).to(self.device)
            encoder_out, encoder_hidden = self.encoder(encoder_input, encoder_hidden)
            loss = 0
            for idx, (source, target) in enumerate(zip(sources, targets)):
                # Extracting hidden states(h,c) for the current item in the batch
                decoder_hidden = [encoder_hidden[0][:, idx:idx + 1].contiguous(),
                                  encoder_hidden[1][:, idx: idx + 1].contiguous()]
                decoder_input = torch.LongTensor([[self.start_token_index]]).to(self.device)
                predicted_indexes = []
                for target_idx in target:
                    if target_idx == self.end_token_index:
                        break
                    decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    predicted_target = decoder_out.argmax(dim=2)
                    actual_target = torch.LongTensor([[target_idx]]).to(self.device)
                    # teacher forcing
                    if np.random.random() < self.teacher_forcing_prob:
                        decoder_input = actual_target
                    else:
                        decoder_input = predicted_target
                    loss += F.cross_entropy(decoder_out.squeeze(0), actual_target.flatten(),
                                            ignore_index=self.pad_token_index)
                    predicted_indexes.append(predicted_target.item())
                    if epoch % 5 == 0:
                        # print(f'Epoch:{epoch}, Loss: {loss.item():.5f}')
                        print(f'T{idx}', self.tokenizer.indexes_to_text(target))
                        print(f'P{idx}:', self.tokenizer.indexes_to_text(predicted_indexes))
                        print('----------------------------------')

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            print('epoc:', epoch, ' loss', loss.item())


tokenizer = Tokenizer(max_sequence_length=MAX_SEQUENCE_LENGTH)
sources, targets = zip(*conversation_pair)
tokenizer.fit_on_text(sources + targets)
sources = tokenizer.texts_to_index(sources)
targets = tokenizer.texts_to_index(targets)
encoder = EncoderLSTM(input_size=tokenizer.words_count, hidden_size=LSTM_HIDDEN_SIZE,
                      embeddings_dims=EMBEDDINGS_DIMS).to(
    DEVICE)
decoder = DecoderLSTM(input_size=tokenizer.words_count, hidden_size=LSTM_HIDDEN_SIZE, embeddings_dims=EMBEDDINGS_DIMS,
                      vocab_size=tokenizer.words_count).to(DEVICE)
trainer = TrainingSession(encoder=encoder, decoder=decoder, tokenizer=tokenizer, learning_rate=LEARNING_RATE,
                          teacher_forcing_prob=TEACHER_FORCING_PROB,
                          device=DEVICE)
trainer.train(sources, targets, epochs=20)
