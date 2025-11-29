import argparse
import os
import random
from argparse import Namespace
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from data.coco_dataset_unigram import CocoDatasetKarpathy as CocoDatasetKarpathy
from data.coco_dataloader import CocoDataLoader as CocoDataLoader

from test_suggestion_precision_recall import evaluate_unique_words_on_set

from losses.loss import LabelSmoothingLoss
from optims.radam import RAdam
from utils import language_utils
from utils.args_utils import str2bool, str2list, scheduler_type_choice, optim_type_choice
import math

from utils.ema_saving_utils import load_most_recent_checkpoint, save_last_checkpoint, partially_load_state_dict
from utils.ema import ModelEma

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import functools

print = functools.partial(print, flush=True)


def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m " + \
        str(int(ticks) % 60) + " s"


def train(rank,
          ema,
          train_args,
          path_args,
          ddp_model,
          coco_dataset, data_loader,
          optimizer, sched,
          max_len,
          ddp_sync_port):
    
    MY_IGNORE_INDEX = -1
    loss_xe = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=coco_dataset.caption_word2idx_dict['PAD'])

    def mean_ds(x, dim=None):
        return (
            x.float().mean().type_as(x)
            if dim is None
            else x.float().mean(dim).type_as(x)
        )

    algorithm_start_time = time()
    saving_timer_start = time()
    time_to_save = False
    running_loss = 0
    running_diff_loss = 0
    running_length_loss = 0
    running_time = 0
    already_trained_steps = data_loader.get_num_batches() * data_loader.get_epoch_it() + data_loader.get_batch_it()
    prev_print_iter = already_trained_steps
    num_iter = data_loader.get_num_batches() * train_args.num_epochs
    for it in range(already_trained_steps, num_iter):
        iter_timer_start = time()
        ddp_model.train()

        batch_input_x, batch_target_y, \
            batch_input_x_num_pads, batch_target_y_num_pads, \
            USED_batch_unique_words, USED_batch_unique_words_num_pads, \
            batch_img_idx \
            = data_loader.get_next_batch(verbose=True *
                                                 (((it + 1) % train_args.print_every_iter == 0) or
                                                  (it + 1) % data_loader.get_num_batches() == 0),
                                         get_also_image_idxes=True)
        batch_input_x = batch_input_x.to(rank)
        USED_batch_unique_words = USED_batch_unique_words.to(rank)

        all_diffuse_dict = ddp_model(enc_x=batch_input_x, dec_x=USED_batch_unique_words,
                                     enc_x_num_pads=batch_input_x_num_pads,
                                     dec_x_num_pads=USED_batch_unique_words_num_pads,
                                     diffusion_t=None, apply_log_softmax=False,
                                     mode='diffusion_train')

        diffusion_loss = 0
        for i in range(ddp_model.module.num_diffusion_steps):
            bs, num_ngrams, k_gram, vocab_size = all_diffuse_dict[i]['decoder_output'].shape
            UNPACKED_input = all_diffuse_dict[i]['decoder_output'].reshape(bs, num_ngrams * k_gram, vocab_size)
            UNPACKED_input = UNPACKED_input.transpose(-1, -2)

            UNPACKED_target = all_diffuse_dict[i]['x_0_ignore'].reshape(bs, num_ngrams * k_gram)

            tmp = \
                (
                        (1 - 1 / (all_diffuse_dict[i]['t'] + 1)) * \
                        F.cross_entropy(input=UNPACKED_input,
                                        target=UNPACKED_target,
                                        ignore_index=MY_IGNORE_INDEX, reduction='none').mean(1)
                )
            tmp = mean_ds(all_diffuse_dict[i]['weights'] * tmp)
            diffusion_loss += tmp
        diffusion_loss /= ddp_model.module.num_diffusion_steps

        length_pred_loss = loss_xe(input=all_diffuse_dict[0]['length_pred'],
                                   target=all_diffuse_dict[0]['length_target'])

        loss = diffusion_loss + length_pred_loss
        running_loss += loss.item()
        running_diff_loss += diffusion_loss.item()
        running_length_loss += length_pred_loss.item()
        loss.backward()

        if it % train_args.num_accum == 0:
            optimizer.step()
            ema.update(ddp_model.module)
            optimizer.zero_grad()

        sched.step()

        current_rl = sched.get_last_lr()[0]

        running_time += time() - iter_timer_start
        if (it + 1) % train_args.print_every_iter == 0:
            avg_loss = running_loss / (it + 1 - prev_print_iter)
            avg_diff_loss = running_diff_loss / (it + 1 - prev_print_iter)
            avg_length_loss = running_length_loss / (it + 1 - prev_print_iter)
            tot_elapsed_time = time() - algorithm_start_time
            avg_time_time_per_iter = running_time / (it + 1 - prev_print_iter)
            print('[GPU:' + str(rank) + '] ' + str(round(((it + 1) / num_iter) * 100, 3)) +
                  ' % it: ' + str(it + 1) + ' lr: ' + str(round(current_rl, 12)) +
                  ' n.acc: ' + str(train_args.num_accum) +
                  ' diff loss: ' + str(round(avg_diff_loss, 3)) +
                  ' length loss: ' + str(round(avg_length_loss, 3)) +
                  ' tot loss: ' + str(round(avg_loss, 3)) +
                  ' elapsed: ' + convert_time_as_hhmmss(tot_elapsed_time) +
                  ' sec/iter: ' + str(round(avg_time_time_per_iter, 3)))
            running_loss = 0
            running_diff_loss = 0
            running_length_loss = 0
            running_time = 0
            prev_print_iter = it + 1

        if ((it + 1) % data_loader.get_num_batches() == 0) or ((it + 1) % train_args.eval_every_iter == 0):

            if rank == 0 and (data_loader.get_epoch_it() >= 1):
                print("Validation Evaluation Precision and Recall")
                evaluate_unique_words_on_set(ema.module,
                                             coco_dataset.caption_idx2word_list,
                                             coco_dataset.caption_word2idx_dict,
                                             None,
                                             coco_dataset.get_sos_token_idx(), coco_dataset.get_eos_token_idx(),
                                             coco_dataset.val_num_images, data_loader,
                                             CocoDatasetKarpathy.ValidationSet_ID, max_len,
                                             rank, ddp_sync_port,
                                             parallel_batches=train_args.eval_parallel_batch_size,
                                             use_images_instead_of_features=False,
                                             beam_sizes=train_args.eval_beam_sizes)

            time_to_save = True

        # saving
        elapsed_minutes = (time() - saving_timer_start) / 60
        if time_to_save or elapsed_minutes > train_args.save_every_minutes or ((it + 1) == num_iter):
            saving_timer_start = time()
            time_to_save = False
            if rank == 0:
                save_last_checkpoint(ema, ddp_model.module, optimizer, sched,
                                     data_loader, path_args.save_path,
                                     num_max_checkpoints=train_args.how_many_checkpoints,
                                     additional_info='xe')


def distributed_train(rank,
                      world_size,
                      model_args,
                      optim_args,
                      coco_dataset,
                      array_of_init_seeds,
                      model_max_len,
                      train_args,
                      path_args):
    print("GPU: " + str(rank) + "][SUGGESTION MODULE TRAINING] Process " + str(rank) + " working...")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = train_args.ddp_sync_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


    from models.suggestion_module.sst_sugg_module import SST_Sugg_Module
    model = SST_Sugg_Module(
        k_gram=train_args.selected_n_gram,
        d_model=model_args.model_dim,
        ff=model_args.model_dim * 4, num_heads=8,
        num_layers=model_args.N_enc, drop_args=model_args.drop_args,
        output_word2idx=coco_dataset.caption_word2idx_dict,
        output_idx2word=coco_dataset.caption_idx2word_list,
        max_seq_len=model_max_len, img_feature_dim=1536,
        num_diffusion_steps=20, rank=rank)

    print("Prediction model's class: " + str(model.__class__))
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    image_size = 384
    data_loader = CocoDataLoader(coco_dataset=coco_dataset,
                                 batch_size=train_args.batch_size,
                                 num_procs=world_size,
                                 array_of_init_seeds=array_of_init_seeds,
                                 dataloader_mode='caption_wise',
                                 resize_image_size=image_size,
                                 rank=rank,
                                 verbose=True)
    ema = ModelEma(model, 0.999)

    base_lr = 1.0
    if optim_args.optim_type == 'radam':
        optimizer = RAdam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr,
                          betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=base_lr)

    if optim_args.sched_type == 'annealing':
        sched_func = lambda it: (min(it, optim_args.warmup_iters) / optim_args.warmup_iters) * \
                                optim_args.lr * (0.8 ** (
                    it // (optim_args.anneal_every_epoch * data_loader.get_num_batches())))
    else:
        num_batches = data_loader.get_num_batches()
        sched_func = lambda it: max((it >= optim_args.warmup_iters) * optim_args.min_lr,
                                    (optim_args.lr / (max(optim_args.warmup_iters - it, 1))) * \
                                    (pow(optim_args.anneal_coeff, it // (num_batches * optim_args.anneal_every_epoch)))
                                    )

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)

    
    if path_args.save_path is not None:
        _, additional_info = load_most_recent_checkpoint(ema, ddp_model.module, optimizer, sched,
                                                         data_loader, rank, path_args.save_path)
    
    changed_batch_size = data_loader.get_batch_size() != train_args.batch_size
    if changed_batch_size:
        if changed_batch_size:
            print("New requested batch size differ from previous checkpoint", end=" ")
            print("- Proceed to reset training session keeping pre-trained weights")
            data_loader.change_batch_size(batch_size=train_args.batch_size, verbose=True)
        
        if optim_args.optim_type == 'radam':
            optimizer = RAdam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=1,
                              betas=(0.9, 0.98), eps=1e-9)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, ddp_model.parameters()), lr=1)

        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_func)

    train(rank,
          ema,
          train_args,
          path_args,
          ddp_model,
          coco_dataset, data_loader,
          optimizer, sched,
          model_max_len,
          train_args.ddp_sync_port)

    print("[GPU: " + str(rank) + " ] Suggestion Module Training Completed! - Closing...")
    dist.destroy_process_group()


def spawn_train_processes(model_args,
                          optim_args,
                          coco_dataset,
                          train_args,
                          path_args
                          ):
    max_sequence_length = coco_dataset.max_seq_len + 20
    print("Max sequence length: " + str(max_sequence_length))
    print("y vocabulary size: " + str(len(coco_dataset.caption_word2idx_dict)))

    world_size = torch.cuda.device_count()
    print("Using - ", world_size, " processes / GPUs!")
    assert (train_args.num_gpus <= world_size), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(train_args.num_gpus))

    array_of_init_seeds = [random.random() for _ in range(train_args.num_epochs * 2)]
    mp.spawn(distributed_train,
             args=(train_args.num_gpus,
                   model_args,
                   optim_args,
                   coco_dataset,
                   array_of_init_seeds,
                   max_sequence_length,
                   train_args,
                   path_args,
                   ),
             nprocs=train_args.num_gpus,
             join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Suggestion Module Training')
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

    parser.add_argument('--selected_n_gram', type=int, default=1,
                        help='n_gram...')

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

    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs before termination.')

    parser.add_argument('--captions_path', type=str, default='./github_ignore_material/raw_data/',
                        help='Location of groudtruth captions.')

    parser.add_argument('--images_path', type=str, default="./github_ignore_material/raw_data/",
                        help='Path of the images')
    parser.add_argument('--features_path', type=str, default="./github_ignore_material/raw_data/",
                        help='Path for the hdf5 file containing backbones output features.')

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
                           drop_args=drop_args)
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
                          features_path=args.features_path
                          )

    train_args = Namespace(batch_size=args.batch_size,
                           num_accum=args.num_accum,
                           num_gpus=args.num_gpus,
                           ddp_sync_port=args.ddp_sync_port,
                           save_every_minutes=args.save_every_minutes,
                           how_many_checkpoints=args.how_many_checkpoints,
                           print_every_iter=args.print_every_iter,
                           eval_every_iter=args.eval_every_iter,
                           eval_parallel_batch_size=args.eval_parallel_batch_size,
                           eval_beam_sizes=args.eval_beam_sizes,
                           num_epochs=args.num_epochs,
                           selected_n_gram=args.selected_n_gram)

    print("train batch_size: " + str(args.batch_size))
    print("num_accum: " + str(args.num_accum))
    print("ddp_sync_port: " + str(args.ddp_sync_port))
    print("save_path: " + str(args.save_path))
    print("num_gpus: " + str(args.num_gpus))

    coco_dataset = CocoDatasetKarpathy(
        k_gram=train_args.selected_n_gram,
        images_path=path_args.images_path,
        coco_annotations_path=path_args.captions_path + "dataset_coco.json",
        preproc_images_hdf5_filepath=None,
        precalc_features_hdf5_filepath=path_args.features_path,
        limited_num_train_images=None,
        limited_num_val_images=5000)

    # train base model
    spawn_train_processes(model_args=model_args,
                          optim_args=optim_args,
                          coco_dataset=coco_dataset,
                          train_args=train_args,
                          path_args=path_args
                          )
