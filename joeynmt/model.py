# coding: utf-8
"""
Module to represents whole models
"""
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder, CriticDecoder, CriticTransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.vocabulary import Vocabulary
from joeynmt.helpers import ConfigurationError, log_peakiness, join_strings
from joeynmt.metrics import bleu


class RewardRegressionModel(nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.l1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.l2=nn.Linear(H, D_out)

    def forward(self, X):
        return self.l2(self.relu(self.l1(X)))

class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super().__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        self._loss_function = None # set by the TrainManager

    @property
    def loss_function(self):
        return self._x

    @loss_function.setter
    def loss_function(self, loss_function: Callable):
        self._loss_function = loss_function

    def reinforce(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
            src_length: Tensor, temperature: float, topk: int, log_probabilities: False, pickle_logs:False):

        """ Computes forward pass for Policy Gradient aka REINFORCE
        
        Encodes source, then step by step decodes and samples token from output distribution.
        Calls the loss function to compute the BLEU and loss

        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param topk: consider top-k parameters for logging
        :param log_probabilities: log probabilities
        :return: loss, logs
        """

        encoder_output, encoder_hidden = self._encode(src, src_length,
            src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        trg_mask = src_mask.new_ones([1, 1, 1])
        distributions = []
        log_probs = 0
        # init hidden state in case of using rnn decoder  
        hidden = self.decoder._init_hidden(encoder_hidden) \
            if hasattr(self.decoder,'_init_hidden') else 0
        attention_vectors = None
        finished = src_mask.new_zeros((batch_size)).byte()
        # decode tokens
        for _ in range(max_output_length):
            previous_words = ys[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys
            logits, hidden, _, attention_vectors = self.decoder(
                trg_embed=self.trg_embed(previous_words),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=hidden,
                prev_att_vector=attention_vectors,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            next_word = distrib.sample()
            log_probs += distrib.log_prob(next_word)
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            # prevent early stopping in decoding when logging gold token
            if not pickle_logs:
                # check if previous symbol was <eos>
                is_eos = torch.eq(next_word, self.eos_index)
                finished += is_eos
                # stop predicting if <eos> reached for all elements in batch
                if (finished >= 1).sum() == batch_size:
                    break
        ys = ys[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        # get reinforce loss
        batch_loss, rewards, old_bleus = self.loss_function(predicted_strings, gold_strings,  log_probs)
        return (batch_loss, log_peakiness(self.pad_index, self.trg_vocab, topk, distributions,
        trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, old_bleus)) \
        if log_probabilities else (batch_loss, [])
        
    def mrt(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor, src_length: Tensor, 
            temperature: float, samples: int, alpha: float, topk: int, add_gold=False, log_probabilities=False, pickle_logs=False):
        """ Computes forward pass for MRT
        
        Encodes source, samples multiple output sequences.
        Coputes rewards and MRT-loss

        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param samples: number of sampled sentences for MRT
        :param alpha: smootheness of MRT
        :param topk: consider top-k parameters for logging
        :param add_gold: add gold translation
        :param log_probabilities: log probabilities
        :return: loss, probability logs
        """
        if add_gold:
            samples = samples+1
        encoder_output, encoder_hidden = self._encode(src, src_length,
                    src_mask)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0)
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        trg_mask = src_mask.new_ones([1, 1, 1])
        total_prob = 0
        distributions = []
        attention_vectors = None
        encoder_output = encoder_output.repeat(samples,1,1)
        if hasattr(self.decoder,'_init_hidden'):
            hidden = self.decoder._init_hidden(encoder_hidden)
            if len(hidden)==2:
                hidden = (hidden[0].repeat(1,samples,1), hidden[1].repeat(1,samples,1)) 
            else: 
                hidden = hidden.repeat(1,samples,1)
        else:
            hidden = (0,0)
        # repeat tensor for vectorized solution
        ys = ys.repeat(samples, 1)
        src_mask = src_mask.repeat(samples,1,1)
        finished = src_mask.new_zeros((batch_size*samples)).byte()
        # decode tokens
        for i in range(max_output_length):
            previous_words = ys[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys
            logits, hidden, _, attention_vectors = self.decoder(
                trg_embed=self.trg_embed(previous_words),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=hidden,
                prev_att_vector=attention_vectors,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            next_word = distrib.sample()
            if add_gold:
                if i < trg.shape[1]:
                    ith_column = trg[:,i]
                else:
                    tensor = torch.ones((batch_size,), dtype=torch.int64)
                    data = [self.pad_index]*batch_size
                    ith_column = tensor.new_tensor(data)
                next_word[-batch_size:] = ith_column
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            total_prob += distrib.log_prob(next_word)
            # prevent early stopping in decoding when logging gold token
            if not pickle_logs:
                # check if previous symbol was <eos>
                is_eos = torch.eq(next_word, self.eos_index)
                finished += is_eos
                # stop predicting if <eos> reached for all elements in batch
                if (finished >= 1).sum() == batch_size*samples:
                    break
        ys = ys[:, 1:]
        all_sequences = torch.stack(torch.split(ys, batch_size))
        sentence_probabs= list(torch.split(total_prob, batch_size))    
        predicted_outputs = [self.trg_vocab.arrays_to_sentences(arrays=sequ,
                                                        cut_at_eos=True) for sequ in all_sequences]
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_sentences = [[join_strings(wordlist) for wordlist in predicted_output] 
            for predicted_output in predicted_outputs]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        all_gold_sentences = [gold_strings]*samples
        # Simon's trick
        list_of_Qs = torch.softmax(torch.stack(sentence_probabs)*alpha, 0)
        # calculate loss
        batch_loss = 0
        for index, Q in enumerate(list_of_Qs):
            for prediction, gold_ref, Q_iter in zip(predicted_sentences[index], all_gold_sentences[index], Q):
                batch_loss -= bleu([prediction], [gold_ref])*Q_iter
        rewards = [bleu([prediction], [gold_ref]) for prediction, gold_ref in zip(predicted_sentences[-1], all_gold_sentences[-1])]
        # currently unused
        Qs_to_return = [q.tolist() for q in list_of_Qs]
        return (batch_loss, log_peakiness(self.pad_index, self.trg_vocab, topk, distributions, \
            trg, batch_size, max_output_length, gold_strings, predicted_sentences, \
                Qs_to_return, rewards, mrt=True, samples=samples)) \
                if log_probabilities else (batch_loss, [])
    
    def ned_a2c(self, max_output_length, src: Tensor, trg: Tensor, src_mask: Tensor,
                        src_length: Tensor, temperature: float, critic: nn.Module, topk: int, log_probabilities=False, pickle_logs=False):
        """ Computes forward pass for NED-A2C
        
        Encodes source, step by step decodes and samples actor output.
        For each step decodes critic output given actor outputs as target
        Computes actor loss and critic loss

        :param max_output_length: max output length
        :param src: source input
        :param trg: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param temperature: softmax temperature
        :param critic: critic network
        :param topk: consider top-k parameters for logging
        :param log_probabilities: log probabilities
        :return: actor loss, critic loss, actor probability logs
        """

        if max_output_length is None:
            max_output_length = int(max(src_length.cpu().numpy()) * 1.5)
        batch_size = src_mask.size(0) 
        trg_mask = src_mask.new_ones([1, 1, 1])
        # init actor parameters
        encoder_output, encoder_hidden = self._encode(
            src, src_length,
            src_mask)
        hidden = (self.decoder._init_hidden(encoder_hidden)) \
            if hasattr(self.decoder,'_init_hidden') else (0,0)
        attention_vectors = None
        ys = encoder_output.new_full([batch_size, 1], self.bos_index, dtype=torch.long)
        log_probs = 0
        distributions = []
        actor_log_probabs = []
        # init critic parameters
        critic_encoder_output, critic_encoder_hidden = critic._encode(
                src, src_length,
                src_mask)
        critic_hidden = (self.decoder._init_hidden(critic_encoder_hidden)) \
            if hasattr(self.decoder,'_init_hidden') else (0,0)
        critic_logits = []
        critic_sequence = critic_encoder_output.new_full(size=[batch_size, 1], fill_value=self.bos_index, dtype=torch.long)
        critic_attention_vectors = None
        # init dict to track eos
        eos_dict = {i:-1 for i in range(batch_size)}
        finished = src_mask.new_zeros((batch_size)).byte()
        # decode with actor
        for i in range(max_output_length):
            previous_words = ys[:, -1].view(-1, 1) if hasattr(self.decoder,'_init_hidden') else ys
            logits, hidden, _, attention_vectors = self.decoder(
                trg_embed=self.trg_embed(previous_words),
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=hidden,
                prev_att_vector=attention_vectors,
                trg_mask=trg_mask
            )
            logits = logits[:, -1]/temperature
            distrib = Categorical(logits=logits)
            distributions.append(distrib)
            sampled_word = distrib.sample()
            log_probs -= distrib.log_prob(sampled_word)
            ys = torch.cat([ys, sampled_word.unsqueeze(-1)], dim=1)
            actor_log_probabs.append(log_probs)
            sampled_word_list = sampled_word.tolist()
            for index in range(len(sampled_word_list)):
                if sampled_word_list[index] == self.eos_index:
                    if eos_dict[index] == -1:
                        eos_dict[index] = i 
            # decode with critic, using actor as target
            critic_logit, critic_hidden, critic_attention_scores, critic_attention_vectors = critic.decoder(
                trg_embed=self.trg_embed(sampled_word.view(-1,1)),
                encoder_output=critic_encoder_output,
                encoder_hidden=critic_encoder_hidden,
                src_mask=src_mask,
                unroll_steps=1,
                hidden=critic_hidden,
                prev_att_vector=critic_attention_vectors,
                trg_mask=trg_mask
            )
            critic_logits.append(critic_logit)
            critic_distrib =  Categorical(logits = critic_logit.view(-1, critic_logit.size(-1)))
            critic_sample = critic_distrib.sample()
            critic_sequence = torch.cat([critic_sequence, critic_sample.view(-1, 1)], -1)
            # prevent early stopping in decoding when logging gold token
            if not pickle_logs:
                # check if previous symbol was <eos>
                is_eos = torch.eq(sampled_word, self.eos_index)
                finished += is_eos
                # stop predicting if <eos> reached for all elements in batch
                if (finished >= 1).sum() == batch_size:
                    break
        ys = ys[:, 1:]
        critic_sequence = critic_sequence[:, 1:]
        predicted_output = self.trg_vocab.arrays_to_sentences(arrays=ys,
                                                        cut_at_eos=True)
        gold_output = self.trg_vocab.arrays_to_sentences(arrays=trg,
                                                    cut_at_eos=True)
        predicted_strings = [join_strings(wordlist) for wordlist in predicted_output]
        gold_strings = [join_strings(wordlist) for wordlist in gold_output]
        # calculate rewards
        bleu_scores = []
        for prediction, gold_ref in zip(predicted_strings, gold_strings):
            bleu_scores.append(bleu([prediction], [gold_ref]))
        bleu_tensor = torch.FloatTensor(bleu_scores).unsqueeze(1)
        if torch.cuda.is_available():
            bleu_tensor = bleu_tensor.cuda()
        critic_logits_tensor = torch.stack(critic_logits)
        critic_logits_tensor = critic_logits_tensor.squeeze()
        if len(critic_logits_tensor.shape) == 1:
            critic_logits_tensor = critic_logits_tensor.unsqueeze(1)
        for dict_index in eos_dict:
            critic_logits_tensor[eos_dict[dict_index]:,dict_index] = 0
        critic_logits = torch.unbind(critic_logits_tensor)
        rewards = [(bleu_tensor-logit).squeeze(1) for logit in critic_logits]
        # calculate critic loss
        critic_loss = torch.cat([torch.pow(bleu_tensor-logit, 2) for logit in critic_logits]).sum()
        # calculate actor loss
        batch_loss = 0
        for log_prob, critic_logit in zip(actor_log_probabs, critic_logits):
            batch_loss += log_prob.unsqueeze(1)*(bleu_tensor-critic_logit)
        batch_loss = batch_loss.sum()
        return ([batch_loss, critic_loss], log_peakiness(self.pad_index, self.trg_vocab, topk, distributions, trg, batch_size, max_output_length, gold_strings, predicted_strings, rewards, bleu_scores)) \
        if log_probabilities else ([batch_loss, critic_loss], [])

    def forward(self, return_type: str = None, **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """ Interface for multi-gpu

        For DataParallel, We need to encapsulate all model call: model.encode(),
        model.decode(), and model.encode_decode() by model.__call__().
        model.__call__() triggers model.forward() together with pre hooks
        and post hooks, which take care of multi-gpu distribution.

        :param return_type: one of {"loss", "encode", "decode"}
        """
        if return_type is None:
            raise ValueError("Please specify return_type: "
                             "{`loss`, `encode`, `decode`}.")

        return_tuple = (None, None, None, None)
        if return_type == "loss":
            assert self.loss_function is not None
            out, _, _, _ = self._encode_decode(
                src=kwargs["src"],
                trg_input=kwargs["trg_input"],
                src_mask=kwargs["src_mask"],
                src_length=kwargs["src_length"],
                trg_mask=kwargs["trg_mask"])

            # compute log probs
            log_probs = F.log_softmax(out, dim=-1)

            # compute batch loss
            batch_loss = self.loss_function(log_probs, kwargs["trg"])

            # return batch loss
            #     = sum over all elements in batch that are not pad
            return_tuple = (batch_loss, None, None, None)
        elif return_type == "reinforce":
            loss, logging = self.reinforce(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            topk=kwargs['topk'],
            log_probabilities=kwargs["log_probabilities"],
            pickle_logs=kwargs["pickle_logs"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "mrt":
            loss, logging = self.mrt(
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            alpha=kwargs["alpha"],
            samples=kwargs["samples"],
            topk=kwargs['topk'],
            add_gold=kwargs["add_gold"],
            log_probabilities=kwargs["log_probabilities"],
            pickle_logs=kwargs["pickle_logs"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "a2c":
            loss, logging = self.ned_a2c(
            critic=kwargs["critic"],
            src=kwargs["src"],
            trg=kwargs["trg"],
            src_mask=kwargs["src_mask"],
            src_length=kwargs["src_length"],
            max_output_length=kwargs["max_output_length"],
            temperature=kwargs["temperature"],
            topk=kwargs['topk'],
            log_probabilities=kwargs["log_probabilities"],
            pickle_logs=kwargs["pickle_logs"]
            )
            return_tuple = (loss, logging, None, None)

        elif return_type == "encode":
            encoder_output, encoder_hidden = self._encode(
                src=kwargs["src"],
                src_length=kwargs["src_length"],
                src_mask=kwargs["src_mask"])

            # return encoder outputs
            return_tuple = (encoder_output, encoder_hidden, None, None)

        elif return_type == "decode":
            outputs, hidden, att_probs, att_vectors = self._decode(
                trg_input=kwargs["trg_input"],
                encoder_output=kwargs["encoder_output"],
                encoder_hidden=kwargs["encoder_hidden"],
                src_mask=kwargs["src_mask"],
                unroll_steps=kwargs["unroll_steps"],
                decoder_hidden=kwargs["decoder_hidden"],
                att_vector=kwargs.get("att_vector", None),
                trg_mask=kwargs.get("trg_mask", None))

            # return decoder outputs
            return_tuple = (outputs, hidden, att_probs, att_vectors)
        return return_tuple

    # pylint: disable=arguments-differ
    def _encode_decode(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                       src_length: Tensor, trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_length: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self._encode(src=src,
                                                      src_length=src_length,
                                                      src_mask=src_mask)
        unroll_steps = trg_input.size(1)
        return self._decode(encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask, trg_input=trg_input,
                            unroll_steps=unroll_steps,
                            trg_mask=trg_mask)

    def _encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(self.src_embed(src), src_length, src_mask)

    def _decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, trg_input: Tensor,
                unroll_steps: int, decoder_hidden: Tensor = None,
                att_vector: Tensor = None, trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param att_vector: previous attention vector (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            prev_att_vector=att_vector,
                            trg_mask=trg_mask)

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed,
                                    self.trg_embed)


class _DataParallel(nn.DataParallel):
    """ DataParallel wrapper to pass through the model attributes """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None,
                is_critic: bool = False) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
            cfg["encoder"]["hidden_size"], \
            "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                    emb_size=src_embed.embedding_dim,
                                    emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                    emb_size=src_embed.embedding_dim,
                                    emb_dropout=enc_emb_dropout)

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        if is_critic: 
            decoder = CriticTransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = TransformerDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        if is_critic: 
            decoder = CriticDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = RecurrentDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)
    #if not False:
    # tie softmax layer with trg embeddings
    if not is_critic: 
        if cfg.get("tied_softmax", False):
            if trg_embed.lut.weight.shape == \
                    model.decoder.output_layer.weight.shape:
                # (also) share trg embeddings and softmax layer:
                model.decoder.output_layer.weight = trg_embed.lut.weight
            else:
                raise ConfigurationError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer.")
                
    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
