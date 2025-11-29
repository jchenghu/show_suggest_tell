import os
import random
import math
import torch
import numpy as np
import argparse
from argparse import Namespace
from utils.args_utils import str2list, str2bool
import copy
from time import time

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.ema import ModelEma

from data.coco_dataloader import CocoDataLoader
from data.coco_dataset_unigram import CocoDatasetKarpathy
from utils import language_utils
from utils.language_utils import compute_num_pads as compute_num_pads
from eval.eval import COCOEvalCap

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import functools
print = functools.partial(print, flush=True)


def compute_evaluation_loss(loss_function,
                            model,
                            data_set,
                            data_loader,
                            num_samples,
                            sub_batch_size,
                            dataset_split,
                            rank=0,
                            verbose=False):
    model.eval()

    sb_size = sub_batch_size

    tot_loss = 0
    num_sub_batch = math.ceil(num_samples / sb_size)
    tot_num_tokens = 0
    for sb_it in range(num_sub_batch):
        from_idx = sb_it * sb_size
        to_idx = min((sb_it + 1) * sb_size, num_samples)

        sub_batch_input_x, sub_batch_target_y, sub_batch_input_x_num_pads, sub_batch_target_y_num_pads, \
          = data_loader.get_batch_samples(img_idx_batch_list=list(range(from_idx, to_idx)),
                                          dataset_split=dataset_split)
        sub_batch_input_x = sub_batch_input_x.to(rank)
        sub_batch_target_y = sub_batch_target_y.to(rank)

        sub_batch_input_x = sub_batch_input_x
        sub_batch_target_y = sub_batch_target_y
        tot_num_tokens += sub_batch_target_y.size(1)*sub_batch_target_y.size(0) - \
                          sum(sub_batch_target_y_num_pads)
        pred = model(enc_x=sub_batch_input_x,
                     dec_x=sub_batch_target_y[:, :-1],
                     enc_x_num_pads=sub_batch_input_x_num_pads,
                     dec_x_num_pads=sub_batch_target_y_num_pads,
                     apply_softmax=False)
        tot_loss += loss_function(pred, sub_batch_target_y[:, 1:],
                                  data_set.get_pad_token_idx(),
                                  divide_by_non_zeros=False).item()
        del sub_batch_input_x, sub_batch_target_y, pred
        torch.cuda.empty_cache()
    tot_loss /= tot_num_tokens
    if verbose and rank == 0:
        print("Validation Loss on " + str(num_samples) + " samples: " + str(tot_loss))

    return tot_loss


def evaluate_model(ddp_model,
                   diff_model,

                   dataset,
                   data_loader,

                   y_word2idx_list,
                   y_idx2word_list,
                   beam_size, max_seq_len,
                   sos_idx, eos_idx,
                   rank, ddp_sync_port,
                   parallel_batches=16,

                   indexes=[0],
                   dataset_split=CocoDatasetKarpathy.TrainSet_ID,
                   use_images_instead_of_features=False,

                   verbose=True,
                   stanford_model_path="./eval/get_stanford_models.sh"):

    start_time = time()

    diff_beam_search_kwargs = {'beam_size': 3,
                          'beam_max_seq_len': max_seq_len,
                          'sample_or_max': 'max',
                          'how_many_outputs': 1,
                          'sos_idx': dataset.get_sos_token_idx(),
                          'eos_idx': dataset.get_eos_token_idx()}

    sub_list_predictions = []
    validate_y = []
    num_samples = len(indexes)

    ddp_model.eval()
    with torch.no_grad():
        sb_size = parallel_batches
        num_iter_sub_batches = math.ceil(len(indexes) / sb_size)
        for sb_it in range(num_iter_sub_batches):
            last_iter = sb_it == num_iter_sub_batches - 1
            if last_iter:
                from_idx = sb_it * sb_size
                to_idx = num_samples
            else:
                from_idx = sb_it * sb_size
                to_idx = (sb_it + 1) * sb_size

            if use_images_instead_of_features:
                sub_batch_x = [data_loader.get_images_by_idx(i, dataset_split=dataset_split, transf_mode='test').unsqueeze(0)
                         for i in list(range(from_idx, to_idx))]
                sub_batch_x = torch.cat(sub_batch_x).to(rank)
                sub_batch_x_num_pads = [0] * sub_batch_x.size(0)
            else:
                sub_batch_x = [data_loader.get_vis_features_by_idx(i, dataset_split=dataset_split)
                         for i in list(range(from_idx, to_idx))]
                sub_batch_x = torch.nn.utils.rnn.pad_sequence(sub_batch_x, batch_first=True).to(rank)
                sub_batch_x_num_pads = compute_num_pads(sub_batch_x)

            validate_y += [data_loader.get_all_image_captions_by_idx(i, dataset_split=dataset_split) \
                           for i in list(range(from_idx, to_idx))]

            ##########################################################################################

            with torch.no_grad():
                diff_model.eval()

                FINAL_PRED, _ = diff_model(enc_x=sub_batch_x,
                                            enc_x_num_pads=sub_batch_x_num_pads,
                                            mode='beam_search', **diff_beam_search_kwargs)
                batch_pred = [pred[0] for pred in FINAL_PRED]

                batch_ngrams = []
                batch_ngrams_num_len = []
                for single_pred in batch_pred:
                    # single_pred: [num_ngrams, k_gram]
                    single_pred_ngram = []
                    batch_ngrams_num_len.append(len(single_pred))
                    for ngram in single_pred:
                        # FACCIO POOLING IN TOKENS UNIVOCI....
                        # if dataset.check_ngram_in_vocab(ngram):
                        if dataset.check_ngram_in_vocab(ngram): # and (not ngram in single_pred_ngram):
                            single_pred_ngram.append(ngram)
                    if len(single_pred_ngram) == 0:
                        single_pred_ngram = [torch.tensor(
                            [dataset.caption_word2idx_dict['PAD'] for _ in range(dataset.k_gram)]),
                                             torch.tensor([dataset.caption_word2idx_dict['PAD'] for _ in
                                                           range(dataset.k_gram)])]
                        single_pred_ngram = torch.cat(single_pred_ngram, dim=0)
                    else:
                        single_pred_ngram = torch.tensor(single_pred_ngram)
                    # [num_ngrams * k_gram]
                    batch_ngrams.append(single_pred_ngram)
                batch_ngrams = torch.nn.utils.rnn.pad_sequence(
                    batch_ngrams, batch_first=True,
                    padding_value=dataset.get_pad_token_idx()).to(rank)
                batch_ngrams_num_pads = [batch_ngrams.size(1) - num_elems for num_elems in batch_ngrams_num_len]
                bs = len(sub_batch_x)
                batch_ngrams = batch_ngrams.reshape(bs, batch_ngrams.size(1) // ddp_model.module.k_gram,
                                                    ddp_model.module.k_gram)
                ############################################################################################

            beam_search_kwargs = {'beam_size': beam_size,
                                  'beam_max_seq_len': max_seq_len,
                                  'sample_or_max': 'max',
                                  'how_many_outputs': 1,
                                  'sos_idx': sos_idx,
                                  'eos_idx': eos_idx}

            output_words, _ = ddp_model(enc_x=sub_batch_x,
                                        enc_x_num_pads=sub_batch_x_num_pads,
                                        sugg_input=batch_ngrams,
                                        sugg_num_pads=batch_ngrams_num_pads,
                                        mode='beam_search', **beam_search_kwargs)

            output_words = [output_words[i][0] for i in range(len(output_words))]

            pred_sentence = language_utils.convert_allsentences_idx2word(output_words, y_idx2word_list)
            for sentence in pred_sentence:
                sub_list_predictions.append(' '.join(sentence[1:-1]))  # remove EOS and SOS

            del sub_batch_x, sub_batch_x_num_pads, output_words

    ddp_model.train()

    if rank == 0 and verbose:
        # dirty code to leave the evaluation code untouched
        list_predictions = [sub_predictions for sub_predictions in sub_list_predictions]
        list_list_references = [[validate_y[i][j] for j in range(len(validate_y[i]))] for i in range(len(validate_y))]

        gts_dict = dict()
        for i in range(len(list_list_references)):
            gts_dict[i] = [{u'image_id': i, u'caption': list_list_references[i][j]}
                           for j in range(len(list_list_references[i]))]

        pred_dict = dict()
        for i in range(len(list_predictions)):
            pred_dict[i] = [{u'image_id': i, u'caption': list_predictions[i]}]

        coco_eval = COCOEvalCap(gts_dict, pred_dict, list(range(len(list_predictions))),
                                get_stanford_models_path=stanford_model_path)
        score_results = coco_eval.evaluate(bleu=True, rouge=True, cider=True, spice=True, meteor=True, verbose=False)
        elapsed_ticks = time() - start_time
        print("Evaluation Phase over " + str(len(validate_y)) + " BeamSize: " + str(beam_size) +
              "  elapsed: " + str(int(elapsed_ticks/60)) + " m " + str(int(elapsed_ticks % 60)) + ' s')
        print(score_results)

    if rank == 0:
        cider_array = score_results[0]
        _, cider_score = cider_array
        return pred_dict, gts_dict, cider_score

    return None, None, None


def evaluate_model_on_set(ddp_model,
                          diff_model,
                          dataset,
                          data_loader,
                          y_word2idx_list,
                          caption_idx2word_list,
                          sos_idx, eos_idx,
                          num_samples,
                          dataset_split,
                          eval_max_len,
                          rank, ddp_sync_port,
                          parallel_batches=16,
                          beam_sizes=[1],
                          stanford_model_path='./eval/get_stanford_models.sh',
                          use_images_instead_of_features=False,
                          get_predictions=False):

    with torch.no_grad():
        ddp_model.eval()

        for beam in beam_sizes:
            pred_dict, gts_dict, cider_score = evaluate_model(ddp_model,
                                                 diff_model,
                                                 dataset,
                                                 data_loader,
                                                 y_word2idx_list,
                                                 y_idx2word_list=caption_idx2word_list,
                                                 beam_size=beam, max_seq_len=eval_max_len,
                                                 sos_idx=sos_idx, eos_idx=eos_idx,
                                                 rank=rank,
                                                 ddp_sync_port=ddp_sync_port,
                                                 parallel_batches=parallel_batches,
                                                 indexes=list(range(num_samples)),
                                                 dataset_split=dataset_split,
                                                 use_images_instead_of_features=use_images_instead_of_features,
                                                 verbose=True,
                                                 stanford_model_path=stanford_model_path)

            if rank == 0 and get_predictions:
                return pred_dict, gts_dict, cider_score

    return None, None, None



#############################################################



def distributed_test(rank,
                      world_size,
                      model_args,
                      coco_dataset,
                      array_of_init_seeds,
                      model_max_len,
                      test_args,
                      path_args):
    print("GPU: " + str(rank) + "][SUGGESTION MODULE TRAINING] Process " + str(rank) + " working...")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = test_args.ddp_sync_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print("TO-DO: set more arguments to accomodate different sugg configurations.")
    from models.suggestion_module.sst_sugg_module import SST_Sugg_Module
    diffusion_model = SST_Sugg_Module(
        k_gram=test_args.selected_n_gram,
        d_model=128,
        ff=128 * 4,
        num_heads=8,
        num_layers=model_args.N_enc, drop_args=model_args.drop_args,
        output_word2idx=coco_dataset.caption_word2idx_dict,
        output_idx2word=coco_dataset.caption_idx2word_list,
        max_seq_len=model_max_len, img_feature_dim=1536,
        num_diffusion_steps=5, rank=rank)

    ema = ModelEma(diffusion_model, 0.999)
    checkpoint = torch.load(path_args.save_sugg_path)
    ema.load_state_dict(checkpoint['ema'])
    print("Loaded suggestion's checkpoint: " + str(path_args.save_sugg_path))
    diffusion_model = ema
    diffusion_model.to(rank)

    from models.prediction.sst_pred_model import SST_ExpNet_Pred
    model = SST_ExpNet_Pred(k_gram=test_args.selected_n_gram,
                            d_model=model_args.model_dim, N_enc=model_args.N_enc,
                            N_dec=model_args.N_dec, num_heads=8, ff=2048,
                            num_exp_enc_list=[32, 64, 128, 256, 512],
                            num_exp_dec=16,
                            output_word2idx=coco_dataset.caption_word2idx_dict,
                            output_idx2word=coco_dataset.caption_idx2word_list,
                            max_seq_len=model_max_len, drop_args=model_args.drop_args,
                            img_feature_dim=1536,
                            rank=rank)

    print("Prediction model's class: " + str(model.__class__))
    model.to(rank)

    caption_checkpoint = torch.load(path_args.save_sst_path)
    model.load_state_dict(caption_checkpoint['model_state_dict'])
    print("Loaded checkpoint: " + str(path_args.save_sst_path))
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    image_size = 384
    data_loader = CocoDataLoader(coco_dataset=coco_dataset,
                                 batch_size=1,
                                 num_procs=world_size,
                                 array_of_init_seeds=array_of_init_seeds,
                                 dataloader_mode='caption_wise',
                                 resize_image_size=image_size,
                                 rank=rank,
                                 verbose=True)

    print("Evaluation on Validation Set")
    evaluate_model_on_set(ddp_model,
          diffusion_model.module,
          coco_dataset,
          data_loader,

          coco_dataset.caption_word2idx_dict,
          coco_dataset.caption_idx2word_list,
          coco_dataset.get_sos_token_idx(),
          coco_dataset.get_eos_token_idx(),
          coco_dataset.val_num_images,
          CocoDatasetKarpathy.ValidationSet_ID, model_max_len,
          rank, test_args.ddp_sync_port,
          parallel_batches=test_args.eval_parallel_batch_size,
          use_images_instead_of_features=False, # test_args.is_end_to_end,
          beam_sizes=test_args.eval_beam_sizes,
          get_predictions=True)

    print("Evaluation on Test Set")
    evaluate_model_on_set(
           ddp_model,
           diffusion_model.module,
           coco_dataset,
           data_loader,

           coco_dataset.caption_word2idx_dict,
           coco_dataset.caption_idx2word_list,
           coco_dataset.get_sos_token_idx(),
           coco_dataset.get_eos_token_idx(),
           coco_dataset.test_num_images,
           CocoDatasetKarpathy.TestSet_ID, model_max_len,
           rank, test_args.ddp_sync_port,
           parallel_batches=test_args.eval_parallel_batch_size,
           use_images_instead_of_features=False, # test_args.is_end_to_end,
           beam_sizes=test_args.eval_beam_sizes)


    print("[GPU: " + str(rank) + " ] Evaluation completed.")
    dist.destroy_process_group()


def spawn_train_processes(model_args,
                          coco_dataset,
                          test_args,
                          path_args
                          ):
    max_sequence_length = coco_dataset.max_seq_len + 20
    print("Max sequence length: " + str(max_sequence_length))
    print("y vocabulary size: " + str(len(coco_dataset.caption_word2idx_dict)))

    world_size = torch.cuda.device_count()
    print("Using - ", world_size, " processes / GPUs!")
    assert (test_args.num_gpus <= world_size), "requested num gpus higher than the number of available gpus "
    print("Requested num GPUs: " + str(test_args.num_gpus))

    array_of_init_seeds = [random.random() for _ in range(10)]
    mp.spawn(distributed_test,
             args=(test_args.num_gpus,
                   model_args,
                   coco_dataset,
                   array_of_init_seeds,
                   max_sequence_length,
                   test_args,
                   path_args,
                   ),
             nprocs=test_args.num_gpus,
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

    parser.add_argument('--selected_n_gram', type=int, default=1,
                        help='n_gram...')

    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs.')
    parser.add_argument('--ddp_sync_port', type=int, default=12354,
                        help='Distributed Data Parallel synchronization port.')
    parser.add_argument('--save_sst_path', type=str, default='./github_ignore_material/saves_suggestion_module/',
                        help='Checkpoint path of the prediction model.')
    parser.add_argument('--save_sugg_path', type=str, default='./github_ignore_material/saves_suggestion_module/',
                        help='Checkpoint path of the suggestion module.')

    parser.add_argument('--eval_parallel_batch_size', type=int, default=16,
                        help='Number of samples to be evaluated in parallel.')
    parser.add_argument('--eval_beam_sizes', type=str2list, default=[3],
                        help='List of Beam Search Widths.')

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

    path_args = Namespace(save_sst_path=args.save_sst_path,
                          save_sugg_path=args.save_sugg_path,
                          images_path=args.images_path,
                          captions_path=args.captions_path,
                          features_path=args.features_path
                          )

    test_args = Namespace(num_gpus=args.num_gpus,
                          ddp_sync_port=args.ddp_sync_port,
                          eval_parallel_batch_size=args.eval_parallel_batch_size,
                          eval_beam_sizes=args.eval_beam_sizes,
                          selected_n_gram=args.selected_n_gram)

    coco_dataset = CocoDatasetKarpathy(
        k_gram=test_args.selected_n_gram,
        images_path=path_args.images_path,
        coco_annotations_path=path_args.captions_path + "dataset_coco.json",
        preproc_images_hdf5_filepath=None,
        precalc_features_hdf5_filepath=path_args.features_path,
        limited_num_train_images=None,
        limited_num_val_images=5000)

    # train base model
    spawn_train_processes(model_args=model_args,
                          coco_dataset=coco_dataset,
                          test_args=test_args,
                          path_args=path_args
                          )