import torch
import numpy as np
import time
import math
from typing import Literal


from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)

class CustomTokenizer:
    def __init__(self, pad_token: str, start_token: str, end_token: str, vocab_size: int = 30000) -> None:
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.pad_token = pad_token
        self.pad_token_idx = None
        self.start_token = start_token
        self.start_token_idx = None
        self.end_token = end_token
        self.end_token_idx = None
        self.vocab_size = vocab_size
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, source) -> None:
        source = (word for sentence in source for word in sentence)
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=[self.pad_token, self.start_token, self.end_token])
        self.tokenizer.train_from_iterator(source, trainer=trainer)
        self.pad_token_idx = self.tokenizer.token_to_id(self.pad_token)
        self.start_token_idx = self.tokenizer.token_to_id(self.start_token)
        self.end_token_idx = self.tokenizer.token_to_id(self.end_token)

    def token_to_id(self, token) -> int | None:
        return self.tokenizer.token_to_id(token)

    def encode(self, source: list[list[str]]) -> list[int]:
        return [self.start_token_idx] + sum([self.tokenizer.encode(s).ids for s in source], []) + [self.end_token_idx]

    def decode(self, encoded: list[int]) -> str:
        decoded = self.tokenizer.decode(encoded, skip_special_tokens=True)
        if len(decoded) > 0 and decoded[0] == ' ':
            return decoded[1:]
        return decoded

    def save(self, filename) -> None:
        self.tokenizer.save(filename)

    def load(self, filename) -> None:
        self.tokenizer = self.tokenizer.from_file(filename)
        self.pad_token_idx = self.tokenizer.token_to_id(self.pad_token)
        self.start_token_idx = self.tokenizer.token_to_id(self.start_token)
        self.end_token_idx = self.tokenizer.token_to_id(self.end_token)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, maxlen: int =5000):
        super().__init__()
        pe = torch.zeros(1, maxlen, d_model)# 1 for the number of the batch

        position = torch.arange(maxlen).unsqueeze(0).unsqueeze(2) # unsqueeze left and right
        # 2n is range(0, d_model, step=2)
        div_term = ( 10000.0 ** (torch.arange(0, d_model, 2)/d_model) ).unsqueeze(0).unsqueeze(0)
        pe[0,:,0::2] = torch.sin(position / div_term) # broadcasting sin will go to the even
        pe[0,:,1::2] = torch.cos(position / div_term) # cos will go to the odd
        self.register_buffer('pe', pe) # register it to the members of the module

    def forward(self, x):
        x = x + self.pe[:,:x.shape[1],:] # (batch_size, seq_len, embedding_dim)
        return x

class MultiHeadAttn(torch.nn.Module):
    def __init__(self, n_head: int, d_model: int, d_keys: int, d_values: int):
        super(MultiHeadAttn, self).__init__()
        self.n_head, self.d_model, self.d_keys, self.d_values = n_head, d_model, d_keys, d_values
        self.scale = 1 / (d_keys ** 0.5) # 1/sqrt(d_k)

        self.Wq_net = torch.nn.Linear(d_model, n_head * d_keys)# d_n -> h x d_k
        self.Wk_net = torch.nn.Linear(d_model, n_head * d_keys)
        self.Wv_net = torch.nn.Linear(d_model, n_head * d_values)# d_n -> h x d_v
        self.Wo_net = torch.nn.Linear(n_head * d_values, d_model)# h * d_v -> d_n

        self.cached_k = None
        self.cached_v = None
        self.cached_attn_out = None

    def forward(self, input, attn_mask = None, optimized_inference=False):
        if optimized_inference:
            return self.optimized_inference_forward(input)
        n_head, d_keys, d_values = self.n_head, self.d_keys, self.d_values

        batch_size = input.shape[0]
        seq_len = input.shape[1]
        # input.shape = (batch_size, seq_len, d_model)
        head_q = self.Wq_net(input)
        q = head_q.view(batch_size, seq_len, n_head, d_keys).transpose(1,2)
        head_k = self.Wk_net(input)
        k = head_k.view(batch_size, seq_len, n_head, d_keys).permute(0,2,3,1)
        head_v = self.Wv_net(input)
        v = head_v.view(batch_size, seq_len, n_head, d_values).transpose(1,2)
        # q.shape = (batch_size, n_head, seq_len, d_keys)
        # k.shape = (batch_size, n_head, d_keys, (encoder)seq_len)
        # v.shape = (batch_size, n_head, (encoder)seq_len, d_values)
        attn_score = torch.matmul(q, k) * self.scale # (batch_size, n_head, (encoder)seq_len, (encoder)seq_len)

        if attn_mask is not None:
            # attn_mask = (seq_len, seq_len)
            # masked_fill places the specified value where we have 1 in the mask
            attn_score = attn_score.masked_fill(attn_mask.unsqueeze(0).unsqueeze(1), -float('inf'))

        attn_prob = torch.nn.functional.softmax(attn_score, dim=3) # (batch_size, n_head, seq_len, seq_len)
        attn_vec = torch.matmul(attn_prob, v) # (batch_size, n_head, seq_len, d_values)
        attn_vec = attn_vec.transpose(1,2).flatten(2,3) # (batch_size, seq_len, n_head * d_values)

        attn_out = self.Wo_net(attn_vec) # (batch_size, seq_len, d_model)
        return attn_out

    def clear_cache(self):
        self.cached_k = None
        self.cached_v = None
        self.cached_attn_out = None

    def optimized_inference_forward(self, input):
        """ The same as forward, but optimized for inference by caching the previous states of k, v and the attention
            making the computation O(seq_len) instead of O(seq_len^2)
        """
        n_head, d_keys, d_values = self.n_head, self.d_keys, self.d_values
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if self.cached_attn_out == None:
            head_q = self.Wq_net(input)
            q = head_q.view(batch_size, seq_len, n_head, d_keys).transpose(1,2)
            head_k = self.Wk_net(input)
            k = head_k.view(batch_size, seq_len, n_head, d_keys).permute(0,2,3,1)
            self.cached_k = k
            head_v = self.Wv_net(input)
            v = head_v.view(batch_size, seq_len, n_head, d_values).transpose(1,2)
            self.cached_v = v
            attn_score = torch.matmul(q, k) * self.scale
            attn_prob = torch.nn.functional.softmax(attn_score, dim=3)
            attn_vec = torch.matmul(attn_prob, v)
            attn_vec = attn_vec.transpose(1,2).flatten(2,3)
            attn_out = self.Wo_net(attn_vec)
            self.cached_attn_out = attn_out
        else:
            # input.shape = (batch_size, seq_len, d_model)
            x = input[:,-1] # (batch_size, d_model)
            new_element_of_head_k = self.Wk_net(x)
            new_element_of_k = torch.unsqueeze(new_element_of_head_k.view(batch_size, n_head, d_keys), dim=3)
            k = torch.cat((self.cached_k, new_element_of_k), dim=3) # (batch_size, n_head, d_keys, seq_len)
            self.cached_k = k
            new_element_of_head_v = self.Wv_net(x)
            new_element_of_v = torch.unsqueeze(new_element_of_head_v.view(batch_size, n_head, d_values), dim=2)
            v = torch.cat((self.cached_v, new_element_of_v), dim=2)
            self.cached_v = v
            # q.shape = (batch_size, n_head, seq_len, d_keys)
            # k.shape = (batch_size, n_head, d_keys, (encoder)seq_len)
            # v.shape = (batch_size, n_head, (encoder)seq_len, d_values)

            new_element_of_head_q = self.Wq_net(x)

            new_element_of_q = torch.unsqueeze(new_element_of_head_q, dim=2).view(batch_size, n_head, 1, d_keys) # (batch_size, n_head, 1, d_keys)
            attn_score_new = torch.matmul(new_element_of_q, k) * self.scale # (batch_size, n_head, 1, (encoder)seq_len)
            attn_prob_new = torch.nn.functional.softmax(attn_score_new, dim=3)

            new_element_of_attn_vec = torch.matmul(attn_prob_new, v) # (batch_size, n_head, 1, d_values)
            new_element_of_attn_vec = new_element_of_attn_vec.transpose(1, 2).flatten(2,3) # (batch_size, 1, n_head x d_values)
            new_element_of_attn_out = self.Wo_net(new_element_of_attn_vec) # (batch_size, 1, d_model)
            attn_out = torch.cat((self.cached_attn_out, new_element_of_attn_out), dim=1)
            self.cached_attn_out = attn_out
        return attn_out

class DecoderCell(torch.nn.Module):
    def __init__(self, n_head, d_model, d_ff, dropout):
        super().__init__()
        self.self_MHA = MultiHeadAttn(n_head, d_model, d_model//n_head, d_model//n_head)

        self.layer_norm_1 = torch.nn.LayerNorm(d_model)
        self.dropout_1 = torch.nn.Dropout(dropout)

        self.W1 = torch.nn.Linear(d_model, d_ff)
        self.W2 = torch.nn.Linear(d_ff, d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model)
        self.dropout_2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None, optimized_inference=False):
        if optimized_inference:
            return self.optimized_inference_forward(x)
        # x.shape = (batch_size, seq_len, d_model)
        # mask.shape = (seq_len, seq_len)
        z1 = self.self_MHA(x, attn_mask = mask)
        z2 = self.layer_norm_1(x + self.dropout_1(z1))

        z3 = self.W2(torch.nn.functional.relu(self.W1(z2)))
        y = self.layer_norm_2(z2 + self.dropout_2(z3))
        return y

    def optimized_inference_forward(self, x):
        z1 = self.self_MHA(x, optimized_inference=True)
        z2 = self.layer_norm_1(x + self.dropout_1(z1))

        z3 = self.W2(torch.nn.functional.relu(self.W1(z2)))
        y = self.layer_norm_2(z2 + self.dropout_2(z3))
        return y

    def clear_cache(self):
        self.self_MHA.clear_cache()

class TransformerLanguageModel(torch.nn.Module):
    def preparePaddedBatch(self, source, tokenizer):
        device = next(self.parameters()).device
        pad_token_idx = tokenizer.pad_token_idx
        sentences = [tokenizer.encode(s) for s in source]

        m = max(len(s) for s in sentences)
        sentences_padded = [ s+(m-len(s))*[pad_token_idx] for s in sentences]
        sentences_padded_tensor = torch.tensor(sentences_padded, dtype=torch.long, device=device)

        return sentences_padded_tensor

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    def __init__(self, file_name: str, n_head: int, d_model: int, d_ff: int, layer_count: int, embed_dropout: float, cell_dropout: float, tokenizer: CustomTokenizer, max_len=5000):
        super(TransformerLanguageModel, self).__init__()
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.embedding = torch.nn.Embedding(self.tokenizer.vocab_size, d_model)

        self.pos_embed = PositionalEncoding(d_model)
        self.dropout = torch.nn.Dropout(embed_dropout)

        self.decoder_cells = torch.nn.ModuleList([DecoderCell(n_head, d_model, d_ff, cell_dropout) for _ in range(layer_count)])

        pos = torch.arange(max_len)
        # where index of the column is larger of the index of the row
        # triangle matrix with true in the right size: (F\T)
        mask = pos.unsqueeze(0)>pos.unsqueeze(1)
        self.register_buffer('mask', mask)
        self.projection = torch.nn.Linear(d_model, self.tokenizer.vocab_size)

    def forward(self, y: list[list[str]]):
        Y = self.preparePaddedBatch(y, self.tokenizer)

        seq_len = Y.shape[1]
        E = self.embedding(Y[:,:-1]) # (batch_size, seq_len-1, d_model)
        Z = self.dropout(self.pos_embed(E))

        # from the mask we take only for our batch
        batch_mask = self.mask[:seq_len-1,:seq_len-1]
        for cell in self.decoder_cells:
            Z = cell(Z, batch_mask)

        Z = self.projection(Z.flatten(0,1))
        Y_bar = Y[:,1:].flatten(0,1)

        H = torch.nn.functional.cross_entropy(Z, Y_bar, ignore_index=self.tokenizer.pad_token_idx)
        return H

    def generate_sentence(self, sentence: str, inference_optimization: bool, inference_strategy: Literal['greedy', 'pure', 'temp', 'nuc', 'beam'], inference_parameter=None, limit: int = 1000):
        if inference_strategy == 'beam':
            return self.generate_sentence_beam(sentence, inference_parameter)

        device = next(self.parameters()).device
        with torch.no_grad():
            result = [self.tokenizer.start_token_idx] + self.tokenizer.encode(sentence)
            while len(result) < limit:
                y = torch.tensor([result], dtype=torch.long, device=device)
                e = self.embedding(y)
                z = self.dropout(self.pos_embed(e))
                seq_len = y.shape[1]
                batch_mask = self.mask[:seq_len,:seq_len]
                for cell in self.decoder_cells:
                    z = cell(z, batch_mask, optimized_inference=inference_optimization)
                z = self.projection(z[0][-1])

                if inference_strategy == 'greedy':
                    wordId = torch.argmax(z).item()
                elif inference_strategy == 'pure':
                    p = torch.nn.functional.softmax(z, dim = 0)
                    wordId = torch.multinomial(p, 1)
                elif inference_strategy == 'temp':
                    p = torch.nn.functional.softmax(z/inference_parameter, dim = 0)
                    wordId = torch.multinomial(p, 1)
                elif inference_strategy == 'nuc':
                    z = torch.nn.functional.softmax(z, dim = 0)
                    sorted_p, sorted_i = torch.sort(z, descending=True)
                    cumulative_sum_p = torch.cumsum(sorted_p, dim=0)
                    sorted_indices_outside_nucleus = (cumulative_sum_p > inference_parameter) # tensor with true and false
                    # shift one right, so at least one element to be false => to not zero
                    sorted_indices_outside_nucleus[1:] = sorted_indices_outside_nucleus[:-1].clone()
                    sorted_indices_outside_nucleus[0] = False
                    indices_outside_nucleus = sorted_i[sorted_indices_outside_nucleus]
                    z[indices_outside_nucleus] = 0
                    wordId = torch.multinomial(z, 1)
                else: assert(False)

                if wordId == self.tokenizer.end_token_idx:
                    break
                else:
                    result.append(wordId)

            for cell in self.decoder_cells:
                cell.clear_cache()

        return self.tokenizer.decode(result[1:])

    def generate_sentence_beam(self, sentence: str, beam_len: int, limit: int = 1000):
        device = next(self.parameters()).device

        with torch.no_grad():
            beam = [([self.tokenizer.start_token_idx] + self.tokenizer.encode(sentence), 0)] * beam_len
            for _ in range(limit):
                new_candidates = []
                if all(candidate[0][-1] == self.tokenizer.end_token_idx for candidate in beam):
                    break
                for candidate, score in beam:
                    if candidate[-1] == self.tokenizer.end_token_idx:
                        continue
                    y = torch.tensor([candidate], dtype=torch.long, device=device)
                    e = self.embedding(y)
                    z = self.dropout(self.pos_embed(e))
                    seq_len = y.shape[1]
                    batch_mask = self.mask[:seq_len,:seq_len]
                    for cell in self.decoder_cells:
                        z = cell(z, batch_mask, optimized_inference=False)
                    z = self.projection(z[0][-1])
                    z_scores = torch.nn.functional.log_softmax(z, dim=-1)
                    for token_score, token_index in zip(*torch.topk(z_scores, beam_len, dim=-1)):
                        new_score = score + token_score.item()

                        length_normalization_factor = len(candidate) + 1
                        normalized_score = new_score / length_normalization_factor

                        if normalized_score > -float('inf'):
                            new_candidate = candidate + [token_index.item()]
                            new_candidates.append((new_candidate, normalized_score))
                new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_len]
                beam = new_candidates
        return self.tokenizer.decode(beam[0][0][1:-1])

    def perplexity(self, test: list[list[str]], batch_size: int):
        testSize = len(test)
        H = 0.
        c = 0
        for b in range(0, testSize, batch_size):
            batch = test[b:min(b+batch_size, testSize)]
            l = sum(len(s)-1 for s in batch)
            c += l
            with torch.no_grad():
                H += l * self(batch)
        return math.exp(H/c)

class Trainer():
    def __init__(
            self,
            vocab_size: int,
            batch_size: int,
            max_epochs: int,
            learning_rate: float = 0.0001,
            clip_grad: float = 5.0,
            learning_rate_decay: float = 0.5,
            log_every: int = 10,
            test_every: int = 1000,
            max_patience: int = 4,
            max_trials: int = 5,
        ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad
        self.learning_rate_decay = learning_rate_decay
        self.log_every = log_every
        self.test_every = test_every
        self.max_patience = max_patience
        self.max_trials = max_trials


    def train(self, model: TransformerLanguageModel, corpus: list[list[str]], dev: list[list[str]], extra_train: bool = False):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        bestPerplexity = math.inf
        learning_rate = self.learning_rate

        if extra_train:
            (bestPerplexity, learning_rate, osd) = torch.load(model.file_name + '.optim')
            print('Best model perplexity: ', bestPerplexity)
            print('Learning rate: ', learning_rate)
            optimizer.load_state_dict(osd)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        idx = np.arange(len(corpus), dtype='int32')
        model.train()
        trial = 0
        patience = 0
        iter = 0
        beginTime = time.time()
        for epoch in range(self.max_epochs):
            np.random.shuffle(idx)
            words = 0
            trainTime = time.time()
            for b in range(0, len(idx), self.batch_size):
                iter += 1
                batch = [ corpus[i] for i in idx[b:min(b+self.batch_size, len(idx))] ]
                batch = sorted(batch, key=lambda e: len(e), reverse=True)

                words += sum( len(s)-1 for s in batch )
                H = model(batch)
                optimizer.zero_grad()
                H.backward()
                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
                optimizer.step()
                if iter % self.log_every == 0:
                    print("Iteration:", iter, "Epoch:", epoch+1, '/', self.max_epochs,", Batch:", b // self.batch_size+1, '/', len(idx) // self.batch_size+1, ", loss: ", H.item(), "words/sec:", words / (time.time() - trainTime), "time elapsed:", (time.time() - beginTime))
                    trainTime = time.time()
                    words = 0
                if iter % self.test_every == 0:
                    model.eval()
                    currentPerplexity = model.perplexity(dev, self.batch_size)
                    model.train()
                    print('Current model perplexity: ',currentPerplexity)

                    if currentPerplexity < bestPerplexity:
                        patience = 0
                        bestPerplexity = currentPerplexity
                        print('Saving new best model.')
                        model.save(model.file_name)
                        torch.save((bestPerplexity, learning_rate, optimizer.state_dict()), model.file_name + '.optim')
                    else:
                        patience += 1
                        if patience == self.max_patience:
                            trial += 1
                            if trial == self.max_trials:
                                print('Early stop!')
                                exit(0)
                            learning_rate *= self.learning_rate_decay
                            print('Load previously best model and decay learning rate to:', learning_rate)
                            model.load(model.file_name)
                            (bestPerplexity,_,osd) = torch.load(model.file_name + '.optim')
                            optimizer.load_state_dict(osd)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate
                            patience = 0

        print('Reached maximum number of epochs!')
        model.eval()
        currentPerplexity = model.perplexity(dev, self.batch_size)
        print('Last model perplexity: ',currentPerplexity)

        if currentPerplexity < bestPerplexity:
            bestPerplexity = currentPerplexity
            print('Saving last model.')
            model.save(model.file_name)
            torch.save((bestPerplexity, learning_rate, optimizer.state_dict()), model.file_name + '.optim')
