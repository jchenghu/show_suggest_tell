import argparse
import os
import random
from argparse import Namespace
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data.coco_dataset_unigram import CocoDatasetKarpathy as CocoDatasetKarpathy
from data.coco_dataloader import CocoDataLoader as CocoDataLoader

from test_captioning import evaluate_model_on_set

from losses.loss import LabelSmoothingLoss
from losses.reward import ReinforceCiderReward
from optims.radam import RAdam
from utils import language_utils
from utils.args_utils import str2bool, str2list, scheduler_type_choice, optim_type_choice
from utils.saving_utils import load_most_recent_checkpoint, save_last_checkpoint, partially_load_state_dict

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import functools
print = functools.partial(print, flush=True)



def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m " + \
        str(int(ticks) % 60) + " s"


def train(rank,
          diffusion_model,
          sugg_coco_dataset,
          train_args,
          path_args,
          ddp_model,
          AR_coco_dataset, AR_data_loader,
          optimizer, sched,
          max_len,
          ddp_sync_port):

    if not train_args.reinforce:
        loss_function = LabelSmoothingLoss(smoothing_coeff=0.1, rank=rank)
        loss_function.to(rank)
    else:  # 'rf'
        num_sampled_captions = 5
        running_logprobs = 0
        running_reward = 0
        running_reward_base = 0

        training_references = AR_coco_dataset.get_all_images_captions(CocoDatasetKarpathy.TrainSet_ID)
        reinforce_reward = ReinforceCiderReward(training_references, AR_coco_dataset.get_eos_token_str(),
                                                num_sampled_captions, rank)

        print("TO-DO: Reinforcement with Suggestions is not implemented yet.")
        exit(-1)


    beam_search_kwargs = {'beam_size': 3,
                          'beam_max_seq_len': max_len,
                          'sample_or_max': 'max',
                          'how_many_outputs': 1,
                          'sos_idx': AR_coco_dataset.get_sos_token_idx(),
                          'eos_idx': AR_coco_dataset.get_eos_token_idx()}

    algorithm_start_time = time()
    saving_timer_start = time()
    time_to_save = False
    running_loss = 0
    running_time = 0
    already_trained_steps = AR_data_loader.get_num_batches() * AR_data_loader.get_epoch_it() + AR_data_loader.get_batch_it()
    prev_print_iter = already_trained_steps
    num_iter = AR_data_loader.get_num_batches() * train_args.num_epochs
    for it in range(already_trained_steps, num_iter):
        iter_timer_start = time()
        ddp_model.train()

        if not train_args.reinforce:
            batch_input_x, batch_target_y, \
            batch_input_x_num_pads, batch_target_y_num_pads, \
                USED_batch_unique_words, USED_batch_unique_words_num_pads, \
                batch_img_idx \
                = AR_data_loader.get_next_batch(verbose=True *
                                                     (((it + 1) % train_args.print_every_iter == 0) or
                                                      (it + 1) % AR_data_loader.get_num_batches() == 0),
                                             get_also_image_idxes=True)
            batch_input_x = batch_input_x.to(rank)
            batch_target_y = batch_target_y.to(rank)

            with torch.no_grad():
                rand_num = random.random()
                if rand_num > 0.50:
                    diffusion_model.eval()

                    FINAL_PRED, _ = diffusion_model(enc_x=batch_input_x,
                                              enc_x_num_pads=batch_input_x_num_pads,
                                              mode='beam_search', **beam_search_kwargs)
                    batch_pred = [pred[0] for pred in FINAL_PRED]

                    batch_ngrams = []
                    batch_ngrams_num_len = []
                    for single_pred in batch_pred:
                        # single_pred: [num_ngrams, k_gram]
                        single_pred_ngram = []
                        for ngram in single_pred:
                            #if AR_coco_dataset.check_ngram_in_vocab(ngram):
                            if AR_coco_dataset.check_ngram_in_vocab(ngram) and (torch.tensor(ngram) not in single_pred_ngram):
                                single_pred_ngram.append(torch.tensor(ngram))
                        batch_ngrams_num_len.append(len(single_pred))
                        if len(single_pred_ngram) == 0:
                            single_pred_ngram = [torch.tensor([AR_coco_dataset.caption_word2idx_dict['PAD'] for _ in range(train_args.selected_n_gram)]),
                                                 torch.tensor([AR_coco_dataset.caption_word2idx_dict['PAD'] for _ in
                                                               range(train_args.selected_n_gram)])]
                            single_pred_ngram = torch.cat(single_pred_ngram, dim=0)
                        else:
                            single_pred_ngram = torch.tensor(single_pred_ngram)
                        # [num_ngrams * k_gram]
                        batch_ngrams.append(single_pred_ngram)
                    batch_ngrams = torch.nn.utils.rnn.pad_sequence(
                        batch_ngrams, batch_first=True,
                        padding_value=AR_coco_dataset.get_pad_token_idx()).to(rank)
                    batch_ngrams_num_pads = [batch_ngrams.size(1) - num_elems for num_elems in batch_ngrams_num_len]
                    bs = len(batch_input_x)
                    batch_ngrams = batch_ngrams.reshape(bs, batch_ngrams.size(1) // train_args.selected_n_gram,
                                                        train_args.selected_n_gram)
                else:
                    batch_ngrams = None
                    batch_ngrams_num_pads = None

            pred_logprobs = ddp_model(enc_x=batch_input_x,
                                      dec_x=batch_target_y[:, :-1],
                                      enc_x_num_pads=batch_input_x_num_pads,
                                      dec_x_num_pads=batch_target_y_num_pads,

                                      sugg_input=batch_ngrams,
                                      sugg_num_pads=batch_ngrams_num_pads,

                                      apply_softmax=False)

            loss = loss_function(pred_logprobs, batch_target_y[:, 1:], AR_coco_dataset.get_pad_token_idx())

            running_loss += loss.item()
            loss.backward()
        else:  # rf mode
            print("TO-DO: Reinforcement with Suggestions is not implemented yet.")
            exit(-1)


        if it % train_args.num_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        sched.step()

        current_rl = sched.get_last_lr()[0]

        running_time += time() - iter_timer_start
        if (it + 1) % train_args.print_every_iter == 0:
            if not train_args.reinforce:
                avg_loss = running_loss / (it+1 - prev_print_iter)
                tot_elapsed_time = time() - algorithm_start_time
                avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
                print('[GPU:' + str(rank) + '] ' + str(round(((it + 1) / num_iter) * 100, 3)) +
                      ' % it: ' + str(it + 1) + ' lr: ' + str(round(current_rl, 12)) +
                      ' n.acc: ' + str(train_args.num_accum) +
                      ' avg loss: ' + str(round(avg_loss, 3)) +
                      ' elapsed: ' + convert_time_as_hhmmss(tot_elapsed_time) +
                      ' sec/iter: ' + str(round(avg_time_time_per_iter, 3)))
                running_loss = 0
                running_time = 0
                prev_print_iter = it + 1
            else:
                avg_loss = running_loss / (it + 1 - prev_print_iter)
                tot_elapsed_time = time() - algorithm_start_time
                avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
                avg_logprobs = running_logprobs / (it + 1 - prev_print_iter)
                avg_reward = running_reward / (it + 1 - prev_print_iter)
                avg_reward_base = running_reward_base / (it + 1 - prev_print_iter)
                print('[GPU:' + str(rank) + '] ' + str(round(((it + 1) / num_iter) * 100, 3)) +
                      ' % it: ' + str(it + 1) + ' lr: ' + str(round(current_rl, 12)) +
                      ' n.acc: ' + str(train_args.num_accum) +
                      ' avg rew loss: ' + str(round(avg_loss, 3)) +
                      ' elapsed: ' + convert_time_as_hhmmss(tot_elapsed_time) +
                      ' sec/iter: ' + str(round(avg_time_time_per_iter, 3)) + '\n'
                      ' avg reward: ' + str(round(avg_reward, 5)) +
                      ' avg base: ' + str(round(avg_reward_base, 5)) +
                      ' avg logprobs: ' + str(round(avg_logprobs, 5)))
                running_loss = 0
                running_time = 0
                running_logprobs = 0
                running_reward = 0
                running_reward_base = 0
                prev_print_iter = it + 1

        MIN_EPOCHS_SAVE_THRESHOLD = 4
        INTERESTING_SCORE_THRESHOLD = 1.244
        if (((it + 1) % AR_data_loader.get_num_batches() == 0) or (it + 1) % train_args.eval_every_iter == 0) \
                and AR_data_loader.get_epoch_it() >= MIN_EPOCHS_SAVE_THRESHOLD:

            if rank == 0:
                print("Evaluation on Validation Set")
                _, _, val_cider_score = evaluate_model_on_set(ddp_model,
                                      diffusion_model,
                                      AR_coco_dataset,
                                      AR_data_loader,

                                      AR_coco_dataset.caption_word2idx_dict,
                                      AR_coco_dataset.caption_idx2word_list,
                                      AR_coco_dataset.get_sos_token_idx(), AR_coco_dataset.get_eos_token_idx(),
                                      AR_coco_dataset.val_num_images,
                                      CocoDatasetKarpathy.ValidationSet_ID, max_len,
                                      rank, ddp_sync_port,
                                      parallel_batches=train_args.eval_parallel_batch_size,
                                      use_images_instead_of_features=train_args.is_end_to_end,
                                      beam_sizes=train_args.eval_beam_sizes,
                                      get_predictions=True)

                if val_cider_score >= INTERESTING_SCORE_THRESHOLD:
                    save_last_checkpoint(ddp_model.module, optimizer, sched,
                                         AR_data_loader, path_args.save_path,
                                         num_max_checkpoints=train_args.how_many_checkpoints,
                                         prefix=str(val_cider_score),
                                         additional_info='rf' if train_args.reinforce else 'xe')
                    print("Saved interesting checkpoint, evaluating test set.")

                    print("Evaluation on Test Set")
                    _, _, test_cider_score = evaluate_model_on_set(ddp_model,
                                          diffusion_model,
                                          AR_coco_dataset,
                                          AR_data_loader,

                                          AR_coco_dataset.caption_word2idx_dict,
                                          AR_coco_dataset.caption_idx2word_list,
                                          AR_coco_dataset.get_sos_token_idx(), AR_coco_dataset.get_eos_token_idx(),
                                          AR_coco_dataset.test_num_images,
                                          CocoDatasetKarpathy.TestSet_ID, max_len,
                                          rank, ddp_sync_port,
                                          parallel_batches=train_args.eval_parallel_batch_size,
                                          use_images_instead_of_features=train_args.is_end_to_end,
                                          beam_sizes=train_args.eval_beam_sizes)



            time_to_save = True


        # saving
        elapsed_minutes = (time() - saving_timer_start) / 60
        if time_to_save or elapsed_minutes > train_args.save_every_minutes or ((it + 1) == num_iter):
            saving_timer_start = time()
            time_to_save = False
            if rank == 0:
                save_last_checkpoint(ddp_model.module, optimizer, sched,
                                     AR_data_loader, path_args.save_path,
                                     num_max_checkpoints=train_args.how_many_checkpoints,
                                     additional_info='rf' if train_args.reinforce else 'xe')


def distributed_train(rank,
                      world_size,
                      model_args,
                      optim_args,
                      AR_coco_dataset,
                      sugg_module_save_path,
                      sugg_coco_dataset,
                      array_of_init_seeds,
                      model_max_len,
                      train_args,
                      path_args):

    print("GPU: " + str(rank) + "] Process " + str(rank) + " working...")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = train_args.ddp_sync_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    img_size = 384
    if train_args.is_end_to_end:
        # TO-DO
        print("TO-DO: Not implemented yet. Please avoid the end-to-end configuration")
        exit(-1)
    else:

        from models.suggestion_module.sst_sugg_module import \
            SST_Sugg_Module
        print("TO-DO: set more arguments to accomodate different sugg configurations.")
        diffusion_model = SST_Sugg_Module(
            k_gram=train_args.selected_n_gram,
            d_model=128,
            ff=128 * 4,
            num_heads=8,
            num_layers=model_args.N_enc, drop_args=model_args.drop_args,
            output_word2idx=AR_coco_dataset.caption_word2idx_dict,
            output_idx2word=AR_coco_dataset.caption_idx2word_list,
            max_seq_len=model_max_len, img_feature_dim=1536,
            num_diffusion_steps=5, rank=rank)

        from utils.ema import ModelEma
        ema = ModelEma(diffusion_model, 0.999)
        sugg_checkpoint = torch.load(sugg_module_save_path)
        ema.load_state_dict(sugg_checkpoint['ema'])
        diffusion_model = ema
        diffusion_model.to(rank)

        print("Diffusion model weights loaded from checkpoint: " + str(sugg_module_save_path))

        from models.prediction.sst_pred_model import SST_ExpNet_Pred
        model = SST_ExpNet_Pred(k_gram=train_args.selected_n_gram,
                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=AR_coco_dataset.caption_word2idx_dict,
                                output_idx2word=AR_coco_dataset.caption_idx2word_list,
                                max_seq_len=model_max_len, drop_args=model_args.drop_args,
                                img_feature_dim=1536,
                                rank=rank)
        print("ExpansionNet model selected...")

    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    print("[NOTE] Since suggestions are not always used we have to force find_unused_parameters, it's the expected behaviour.")
    print(model.__class__)

    if train_args.reinforce:
        print("Reinforcement learning Mode")
        AR_data_loader = CocoDataLoader(coco_dataset=AR_coco_dataset,
                                     batch_size=train_args.batch_size,
                                     num_procs=world_size,
                                     array_of_init_seeds=array_of_init_seeds,
                                     dataloader_mode='image_wise',
                                     resize_image_size=img_size if train_args.is_end_to_end else None,
                                     rank=rank,
                                     verbose=True)
        print("TO-DO: Reinforcement with Suggestions is not implemented yet.")
        exit(-1)
    else:
        print("Cross Entropy learning mode")
        AR_data_loader = CocoDataLoader(coco_dataset=AR_coco_dataset,
                                     batch_size=train_args.batch_size,
                                     num_procs=world_size,
                                     array_of_init_seeds=array_of_init_seeds,
                                     dataloader_mode='caption_wise',
                                     resize_image_size=img_size if train_args.is_end_to_end else None,
                                     rank=rank,
                                     verbose=True)

    base_lr = 1.0
    if optim_args.optim_type == 'radam':
        optimizer = RAdam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr,
                          betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr)

    if optim_args.sched_type == 'annealing':
        sched_func = lambda it: (min(it, optim_args.warmup_iters) / optim_args.warmup_iters) * \
                        optim_args.lr * (0.8 ** (it // (optim_args.anneal_every_epoch * AR_data_loader.get_num_batches())))
    else:  # optim_args.sched_type == 'custom_warmup_anneal':
        num_batches = AR_data_loader.get_num_batches()
        sched_func = lambda it: max((it >= optim_args.warmup_iters) * optim_args.min_lr,
                                    (optim_args.lr / (max(optim_args.warmup_iters - it, 1))) * \
                                    (pow(optim_args.anneal_coeff, it // (num_batches * optim_args.anneal_every_epoch)))
                                    )

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)

    if path_args.backbone_save_path != '' or path_args.body_save_path != '':
        if train_args.is_end_to_end:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(path_args.backbone_save_path, map_location=map_location)
            if 'model' in checkpoint.keys():
                partially_load_state_dict(model.swin_transf, checkpoint['model'])
            elif 'model_state_dict' in checkpoint.keys():
                partially_load_state_dict(model, checkpoint['model_state_dict'])
            print("Backbone loaded...", end=' ')
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(path_args.body_save_path, map_location=map_location)
            partially_load_state_dict(model, checkpoint['model_state_dict'])
            print("Body loaded")
        else:
            if train_args.partial_load:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                checkpoint = torch.load(path_args.body_save_path, map_location=map_location)
                partially_load_state_dict(model, checkpoint['model_state_dict'])
                print("Partial load done.")
    else:
        change_from_xe_to_rf = False
        if path_args.save_path is not None:
            _, additional_info = load_most_recent_checkpoint(ddp_model.module, optimizer, sched,
                                                             AR_data_loader, rank, path_args.save_path)
            if additional_info == 'xe' and train_args.reinforce:
                change_from_xe_to_rf = True
            else:
                print("Training mode still in the same stage: " + additional_info)

        changed_batch_size = AR_data_loader.get_batch_size() != train_args.batch_size
        if changed_batch_size or change_from_xe_to_rf:
            if changed_batch_size:
                print("New requested batch size differ from previous checkpoint", end=" ")
                print("- Proceed to reset training session keeping pre-trained weights")
                AR_data_loader.change_batch_size(batch_size=train_args.batch_size, verbose=True)
            else:  # change_from_xe_to_rf
                print("Passing from XE training to RL - Optimizer and data loader states are resetted.")
                AR_data_loader.set_epoch_it(epoch=0, verbose=True)

            if optim_args.optim_type == 'radam':
                optimizer = RAdam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=1,
                                  betas=(0.9, 0.98), eps=1e-9)
            else:
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=1)

            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)

    train(rank,
          diffusion_model.module,
          sugg_coco_dataset,
          train_args,
          path_args,
          ddp_model,
          AR_coco_dataset, AR_data_loader,
          optimizer, sched,
          model_max_len if not train_args.reinforce else train_args.scst_max_len,
          train_args.ddp_sync_port)

    print("[GPU: " + str(rank) + " ] Prediction Model Training Completed " + str( ) + " Closing.")
    dist.destroy_process_group()


def spawn_train_processes(model_args,
                          optim_args,
                          AR_coco_dataset,
                          sugg_module_save_path,
                          sugg_coco_dataset,
                          train_args,
                          path_args
                          ):

    max_sequence_length = AR_coco_dataset.max_seq_len + 20
    print("Max sequence length: " + str(max_sequence_length))
    print("y vocabulary size: " + str(len(AR_coco_dataset.caption_word2idx_dict)))

    world_size = torch.cuda.device_count()
    print("Using - ", world_size, " processes / GPUs!")
    assert(train_args.num_gpus <= world_size), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(train_args.num_gpus))

    array_of_init_seeds = [random.random() for _ in range(train_args.num_epochs*2)]
    mp.spawn(distributed_train,
             args=(train_args.num_gpus,
                   model_args,
                   optim_args,
                   AR_coco_dataset,
                   sugg_module_save_path,
                   sugg_coco_dataset,
                   array_of_init_seeds,
                   max_sequence_length,
                   train_args,
                   path_args,
                   ),
             nprocs=train_args.num_gpus,
             join=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--model', type=str, default='expansionnet',
                        help='Model Class.')
    parser.add_argument('--model_dim', type=int, default=512,
                        help='Model dimension.')
    parser.add_argument('--N_enc', type=int, default=3,
                        help='Number of encoder layers.')
    parser.add_argument('--N_dec', type=int, default=3,
                        help='Number of decoder layers.')
    parser.add_argument('--enc_drop', type=float, default=0.1,
                        help='Dropout percentage in the encoder.')
    parser.add_argument('--dec_drop', type=float, default=0.1,
                        help='Dropout percentage in the decoder.')
    parser.add_argument('--enc_input_drop', type=float, default=0.1,
                        help='Dropout percentage in the visual projection.')
    parser.add_argument('--dec_input_drop', type=float, default=0.1,
                        help='Dropout percentage in the text embeddings.')
    parser.add_argument('--drop_other', type=float, default=0.1,
                        help='Default argument of dropout for remaining elements.')

    parser.add_argument('--optim_type', type=optim_type_choice, default='adam',
                        help='Optimizer type.')
    parser.add_argument('--sched_type', type=scheduler_type_choice, default='fixed',
                        help='Scheduler type.')

    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Initial learning rate.')
    parser.add_argument('--min_lr', type=float, default=5e-7,
                        help='Minimum learning rate.')
    parser.add_argument('--warmup_iters', type=int, default=4000,
                        help='Number of warm-up steps.')
    parser.add_argument('--anneal_coeff', type=float, default=0.8,
                        help='Annealing coefficient.')
    parser.add_argument('--anneal_every_epoch', type=float, default=3.0,
                        help='Annealing period in epochs.')
    parser.add_argument('--sugg_module_save_path', type=str, default='',
                        help='diffusion save path')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of samples in mini-batch.')
    parser.add_argument('--num_accum', type=int, default=1,
                        help='Number of gradient accumulation for each training step.')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs.')
    parser.add_argument('--ddp_sync_port', type=int, default=12354,
                        help='Distributed Data Parallel synchronization port.')
    parser.add_argument('--save_path', type=str, default='./github_ignore_material/saves_suggestion_module/',
                        help='Checkpoint folder.')
    parser.add_argument('--save_every_minutes', type=int, default=25,
                        help='Time period, in minutes, between checkpoints.')
    parser.add_argument('--how_many_checkpoints', type=int, default=1,
                        help='Maximum number of checkpoints.')
    parser.add_argument('--print_every_iter', type=int, default=1000,
                        help='Printing period expressed in number of forward operations.')

    parser.add_argument('--eval_every_iter', type=int, default=999999,
                        help='Evaluation period, expressed in number of forward operations. Regardless of this value. Evaluation is performed every epoch')
    parser.add_argument('--eval_parallel_batch_size', type=int, default=16,
                        help='Number of samples to be evaluated in parallel.')
    parser.add_argument('--eval_beam_sizes', type=str2list, default=[3],
                        help='List of Beam Search Widths.')

    parser.add_argument('--reinforce', type=str2bool, default=False,
                        help='A toggle for reinforcement and cross-entropy learning.')
    parser.add_argument('--scst_max_len', type=int, default=20,
                        help='Maximum sequence length of captions sampled during reinforcement.')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs before termination.')

    parser.add_argument('--captions_path', type=str, default='./github_ignore_material/raw_data/',
                        help='Location of groudtruth captions.')
    parser.add_argument('--partial_load', type=str2bool, default=False,
                        help='Flag for requesting partial loading of the fusion model from an end-to-end model. Not required in case of end-to-end model.')
    parser.add_argument('--backbone_save_path', type=str, default='',
                        help='Path of a checkpoint equipped with backbone.')
    parser.add_argument('--body_save_path', type=str, default='',
                        help='Path of the fusion model-only checkpoint.')
    parser.add_argument('--is_end_to_end', type=str2bool, default=True,
                        help='Toggle for an end-to-end model or fusion model only.')

    parser.add_argument('--sugg_DICT_threshold_vocab', type=int, default=300,
                        help='sugg DICT Threshold vocab...')

    parser.add_argument('--selected_n_gram', type=int, default=1,
                        help='n_gram...')

    parser.add_argument('--images_path', type=str, default="./github_ignore_material/raw_data/",
                        help='Path of the images')
    parser.add_argument('--preproc_images_hdf5_filepath', type=str, default=None,
                        help='Path for the hdf5 file containing preprocessed images.')
    parser.add_argument('--features_path', type=str, default="./github_ignore_material/raw_data/",
                        help='Path for the hdf5 file containing backbones output features.')
    parser.add_argument('--json_pred', type=str, default="",
                        help='Suggestions json')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Training seed.')

    args = parser.parse_args()
    args.ddp_sync_port = str(args.ddp_sync_port)

    # Seed setting ---------------------------------------------
    seed = args.seed
    print("seed: " + str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    drop_args = Namespace(enc=args.enc_drop,
                          dec=args.dec_drop,
                          enc_input=args.enc_input_drop,
                          dec_input=args.dec_input_drop,
                          other=args.drop_other)

    model_args = Namespace(model_dim=args.model_dim,
                           N_enc=args.N_enc,
                           N_dec=args.N_dec,
                           drop_args=drop_args,
                           model=args.model)
    optim_args = Namespace(lr=args.lr,
                           min_lr=args.min_lr,
                           warmup_iters=args.warmup_iters,
                           anneal_coeff=args.anneal_coeff,
                           anneal_every_epoch=args.anneal_every_epoch,
                           optim_type=args.optim_type,
                           sched_type=args.sched_type)

    path_args = Namespace(save_path=args.save_path,
                          images_path=args.images_path,
                          captions_path=args.captions_path,
                          features_path=args.features_path,
                          backbone_save_path=args.backbone_save_path,
                          body_save_path=args.body_save_path,
                          preproc_images_hdf5_filepath=args.preproc_images_hdf5_filepath
                          )

    train_args = Namespace(is_end_to_end=args.is_end_to_end,
                           batch_size=args.batch_size,
                           num_accum=args.num_accum,
                           num_gpus=args.num_gpus,
                           ddp_sync_port=args.ddp_sync_port,
                           save_every_minutes=args.save_every_minutes,
                           how_many_checkpoints=args.how_many_checkpoints,
                           print_every_iter=args.print_every_iter,
                           eval_every_iter=args.eval_every_iter,
                           eval_parallel_batch_size=args.eval_parallel_batch_size,
                           eval_beam_sizes=args.eval_beam_sizes,
                           reinforce=args.reinforce,
                           num_epochs=args.num_epochs,
                           partial_load=args.partial_load,
                           scst_max_len=args.scst_max_len,
                           json_pred=args.json_pred,
                           selected_n_gram=args.selected_n_gram)

    print("train batch_size: " + str(args.batch_size))
    print("num_accum: " + str(args.num_accum))
    print("ddp_sync_port: " + str(args.ddp_sync_port))
    print("save_path: " + str(args.save_path))
    print("num_gpus: " + str(args.num_gpus))

    AR_coco_dataset = CocoDatasetKarpathy(
        k_gram=train_args.selected_n_gram,
        images_path=path_args.images_path,
        coco_annotations_path=path_args.captions_path + "dataset_coco.json",
        preproc_images_hdf5_filepath=path_args.preproc_images_hdf5_filepath if train_args.is_end_to_end else None,
        precalc_features_hdf5_filepath=None if train_args.is_end_to_end else path_args.features_path,
        limited_num_train_images=None,
        limited_num_val_images=5000)

    # train base model
    spawn_train_processes(model_args=model_args,
                          optim_args=optim_args,
                          AR_coco_dataset=AR_coco_dataset,
                          sugg_module_save_path=args.sugg_module_save_path,
                          sugg_coco_dataset=None,
                          train_args=train_args,
                          path_args=path_args
                          )
