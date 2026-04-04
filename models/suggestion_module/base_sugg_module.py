"""
Credits to:
    https://github.com/HKUNLP/reparam-discrete-diffusion
    for the part on reparametrization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import torch.distributions as dists


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class BaseSuggestionModule(nn.Module):
    def __init__(self):
        super(BaseSuggestionModule, self).__init__()
        # mandatory attributes
        # rank: to enable multiprocessing
        self.rank = None
        self.k_gram = None

    def check_required_attributes(self):
        if self.rank is None:
            raise NotImplementedError("Subclass must assign the rank integer according to the GPU group")
        if self.k_gram is None:
            raise NotImplementedError("k gram must be assigned")

    def forward_enc(self, enc_input, enc_input_num_pads):
        raise NotImplementedError

    def forward_dec(self, cross_input, enc_input_num_pads, dec_input, dec_input_num_pads,
                    diffusion_t, apply_log_softmax=False):
        raise NotImplementedError

    def forward(self, enc_x, dec_x=None,
                enc_x_num_pads=[0], dec_x_num_pads=[0],
                diffusion_t=None, apply_log_softmax=False,
                mode='forward', **kwargs):
        if mode == 'diffusion_train':
            x = self.forward_enc(enc_x, enc_x_num_pads)
            diffuse_dict = self.diffusion_train(x, enc_x_num_pads, dec_x, dec_x_num_pads, apply_log_softmax)
            return diffuse_dict
        else:
            assert ('sos_idx' in kwargs.keys() or 'eos_idx' in kwargs.keys()), \
                'sos and eos must be provided in case of batch sampling or beam search'
            sos_idx = kwargs.get('sos_idx', -999)
            eos_idx = kwargs.get('eos_idx', -999)
            if mode == 'beam_search':
                beam_size_arg = kwargs.get('beam_size', 5)
                how_many_outputs_per_beam = kwargs.get('how_many_outputs', 1)
                beam_max_seq_len = kwargs.get('beam_max_seq_len', 20)
                sample_or_max = kwargs.get('sample_or_max', 'max')
                out_classes, out_logprobs = self.beam_search_reparam(
                    enc_x, enc_x_num_pads,
                    beam_size=beam_size_arg,
                    sos_idx=sos_idx,
                    eos_idx=eos_idx,
                    how_many_outputs=how_many_outputs_per_beam,
                    max_seq_len=beam_max_seq_len,
                    sample_or_max=sample_or_max)
                return out_classes, out_logprobs

    @staticmethod
    def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
        """
        scores: [b, n]
        cutoff_len: [b, 1]
        stochastic: bool, whether to add noise to select top_k or not
        returns:
            mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
        """
        if stochastic:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            _scores = scores + temp * gumbel_noise
        else:
            _scores = scores
        sorted_index = _scores.sort(-1)[0]
        cutoff = sorted_index.gather(dim=-1, index=cutoff_len) + 1e-10
        # cutoff_len = k -> select k + 1 tokens
        masking = _scores < cutoff
        return masking

    def _reparam_decoding(self, input_tokens, input_scores, cur_tokens, cur_scores,
                          decoding_strategy, xt_neq_x0, non_special_sym_mask, t, max_step, noise
                          ):
        # output_tokens: [B, N], output_scores: [B, N]
        # cur_tokens: [B, N], cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)

        # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # first set the denoising rate according to the schedule
        if schedule == "linear":
            rate = 1 - (t + 1) / max_step
        elif schedule == "cosine":
            rate = np.cos((t + 1) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # compute the cutoff length for denoising top-k positions
        cutoff_len = (
                non_special_sym_mask.sum(1, keepdim=True).type_as(input_scores) * rate
        ).long()
        # set the scores of special symbols to a large value so that they will never be selected
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)

        # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = self.topk_masking(_scores_for_topk, cutoff_len, stochastic=True,
                                              temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = self.topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError

        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.

        # #### the `uncond` mode ####
        # In our reparameterized decoding,
        # both v1_t and v2_t can be fully determined by the current token scores .

        # #### the `cond` mode ####
        # However, we can also impose some conditional constraints on v1_t so that
        # the decoding can be performed in a more conservative manner.
        # For example, we can set v1_t = 0 only when
        # (the newly output tokens are the same as previous denoised results, AND
        # the current token score becomes lower, AND
        # the current token score is not in the top-k share among all tokens).
        if condition == "cond":
            not_v1_t = (cur_tokens == input_tokens) & (cur_scores < input_scores) & lowest_k_mask
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError

        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)

        next_tokens = input_tokens.masked_fill(masked_to_noise, noise)
        # I don't want to change it here, less clear
        next_scores = input_scores.masked_fill(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        next_tokens = next_tokens.masked_scatter(masked_to_x0, cur_tokens[masked_to_x0])
        next_scores = next_scores.masked_scatter(masked_to_x0, cur_scores[masked_to_x0])

        # b_{t} = (b_{t+1} & u_t) | v_t
        # For convenience, save the NOT of b_t for the next iteration
        # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        return new_xt_neq_x0, next_tokens, next_scores

    def beam_search_reparam(self, enc_input, enc_input_num_pads, sos_idx, eos_idx,
                            beam_size=3, how_many_outputs=1, max_seq_len=20, sample_or_max='max', ):
        bs = enc_input.shape[0]
        device = enc_input.device

        HOW_MANY_LENGTHS = 3
        HOW_MANY_OUTPUTS = 1
        TEMPERATURE = 1.0

        cross_enc_output = self.forward_enc(enc_input, enc_input_num_pads)

        # [bs, enc_len, d_model]
        length_pred = self.forward_length(cross_enc_output, enc_input_num_pads)
        length_pred = F.softmax(length_pred, dim=-1)
        # [bs, max_pred]
        _, all_length_pred = torch.topk(length_pred, k=HOW_MANY_LENGTHS, dim=-1)

        # repeat according to how_many_outputs
        all_length_pred = all_length_pred.unsqueeze(1).repeat(1, HOW_MANY_OUTPUTS, 1). \
            reshape(bs * HOW_MANY_OUTPUTS, HOW_MANY_LENGTHS)
        cross_enc_output = cross_enc_output.unsqueeze(1).repeat(1, HOW_MANY_OUTPUTS, 1, 1). \
            reshape(bs * HOW_MANY_OUTPUTS, cross_enc_output.size(1), cross_enc_output.size(2))
        repeated_bs = bs * HOW_MANY_OUTPUTS
        enc_input_num_pads = [enc_input_num_pads[i] for i in range(bs) for _ in range(HOW_MANY_OUTPUTS)]

        all_predictions = []
        all_logprobs = []
        for l in range(HOW_MANY_LENGTHS):

            length_pred = all_length_pred[:, l]

            # intial tokens
            max_length_in_batch = torch.max(length_pred).item()
            dec_input_num_pads = []
            for pred_len in length_pred.tolist():
                dec_input_num_pads.append(max_length_in_batch - pred_len)

            lengths_pred_tmp = length_pred.unsqueeze(1).repeat(1, max_length_in_batch)
            mask_tmp = torch.ones(size=(repeated_bs, max_length_in_batch), dtype=torch.long, device=device) * \
                       self.output_word2idx['MASK']
            arange_tmp = torch.arange(max_length_in_batch, device=device).view(1, max_length_in_batch).repeat(
                repeated_bs, 1)
            # [bs, seq_len]
            pad_mask = arange_tmp > lengths_pred_tmp
            init_tokens_batch = mask_tmp.masked_fill(pad_mask, self.output_word2idx['PAD'])
            init_tokens_batch = init_tokens_batch.unsqueeze(-1).repeat(1, 1, self.k_gram)
            # [bs, seq_len, k_gram]

            # initially set to MASK
            input_scores_batch = torch.zeros(repeated_bs, max_length_in_batch, self.k_gram,
                                             device=device)  # , len(self.output_word2idx)
            input_tokens_batch = init_tokens_batch
            MAX_INFERENCE_STEPS = self.num_diffusion_steps  # 50  # 10
            STEP_SIZE = self.num_diffusion_steps // MAX_INFERENCE_STEPS

            # initial xt_neq_x0

            UNPACKED_input_tokens_batch = input_tokens_batch.reshape(
                repeated_bs, max_length_in_batch * self.k_gram
            )
            UNPACKED_curr_xt_neq_x0 = UNPACKED_input_tokens_batch.ne(self.output_word2idx['PAD'])
            UNPACKED_input_scores_batch = input_scores_batch.reshape(repeated_bs, max_length_in_batch * self.k_gram)
            assert (
                    self.num_diffusion_steps % MAX_INFERENCE_STEPS == 0), "Num inference steps must be multiple of max num diffusion"
            for infer_step in range(MAX_INFERENCE_STEPS):
                # note: corresp_diff_step follows a descending order
                # corresp_diff_step = self.num_diffusion_steps - infer_step * STEP_SIZE
                diffusion_step = self.num_diffusion_steps - (infer_step + 1) * STEP_SIZE

                tensor_diffusion_step = (torch.ones(repeated_bs, dtype=torch.long, device=device) * diffusion_step)

                UNPACKED_dec_input_num_pads = dec_input_num_pads  # * 2
                UNPACKED_scores = self.forward_dec(
                    cross_input=cross_enc_output, enc_input_num_pads=enc_input_num_pads,
                    dec_input=UNPACKED_input_tokens_batch, dec_input_num_pads=UNPACKED_dec_input_num_pads,
                    diffusion_t=tensor_diffusion_step,
                    apply_log_softmax=True
                )

                if HOW_MANY_OUTPUTS == 1:
                    UNPACKED_cur_scores, UNPACKED_cur_tokens = UNPACKED_scores.max(-1)
                else:
                    UNPACKED_cur_tokens = dists.Categorical(logits=UNPACKED_scores / TEMPERATURE).sample()
                    UNPACKED_cur_scores = torch.gather(UNPACKED_scores, -1,
                                                       UNPACKED_cur_tokens.unsqueeze(-1)).squeeze(-1)

                reparam_diffusion_step = (infer_step + 1) * STEP_SIZE - 1
                UNPACKED_next_xt_neq_x0, UNPACKED_next_tokens_batch, UNPACKED_next_scores_batch = self._reparam_decoding(
                    input_tokens=UNPACKED_input_tokens_batch, input_scores=UNPACKED_input_scores_batch,
                    cur_tokens=UNPACKED_cur_tokens, cur_scores=UNPACKED_cur_scores,

                    decoding_strategy='strategy-cond-deterministic-linear',
                    xt_neq_x0=UNPACKED_curr_xt_neq_x0,
                    non_special_sym_mask=UNPACKED_input_tokens_batch.ne(self.output_word2idx['PAD']),

                    t=reparam_diffusion_step,
                    max_step=self.num_diffusion_steps,

                    noise=self.output_word2idx['MASK'],
                )

                UNPACKED_curr_xt_neq_x0 = UNPACKED_next_xt_neq_x0
                UNPACKED_input_tokens_batch = UNPACKED_next_tokens_batch
                UNPACKED_input_scores_batch = UNPACKED_next_scores_batch

            input_scores_batch = UNPACKED_input_scores_batch.reshape(repeated_bs, max_length_in_batch, self.k_gram)
            input_tokens_batch = UNPACKED_input_tokens_batch.reshape(repeated_bs, max_length_in_batch, self.k_gram)

            res_caption_pred = [[] for _ in range(bs)]
            res_caption_logprob = [[] for _ in range(bs)]
            for i in range(bs):
                q = i * HOW_MANY_OUTPUTS
                max_logprob = -9999999
                max_index = 0
                for j in range(HOW_MANY_OUTPUTS):
                    actual_length = max_length_in_batch - dec_input_num_pads[q + j]
                    score_mean = input_scores_batch[q + j, :actual_length, :].mean()
                    if score_mean >= max_logprob:
                        max_logprob = score_mean
                        max_index = j

                actual_length = max_length_in_batch - dec_input_num_pads[q + max_index]
                res_caption_pred[i].append(input_tokens_batch[q + max_index, :actual_length, :].tolist())
                res_caption_logprob[i].append(input_scores_batch[q + max_index, :actual_length, :])

            all_predictions.append(res_caption_pred)
            all_logprobs.append(res_caption_logprob)

        best_caption_pred = [[] for _ in range(bs)]
        best_caption_logprob = [[] for _ in range(bs)]
        for i in range(bs):
            max_length_index = 0
            max_length_logprob = -9999999
            for j in range(HOW_MANY_LENGTHS):
                sum_logprobs = all_logprobs[j][i][0].mean()
                if sum_logprobs >= max_length_logprob:
                    max_length_index = j
                    max_length_logprob = sum_logprobs
            best_caption_pred[i].append(all_predictions[max_length_index][i][0])
            best_caption_logprob[i].append(all_logprobs[max_length_index][i][0])

        return best_caption_pred, best_caption_logprob
