import os
import random
import math
import torch
import argparse
from argparse import Namespace
from utils.args_utils import str2list, str2bool
import copy
from time import time

import torch.nn.functional as F

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




def evaluate_unique_words(ddp_model,
                   y_idx2word_list,
                   y_word2idx_dict,
                   ternary_word2idx_dict,


                   beam_size, max_seq_len,
                   sos_idx, eos_idx,
                   rank, ddp_sync_port,
                   parallel_batches=16,

                   indexes=[0],
                   data_loader=None,
                   dataset_split=CocoDatasetKarpathy.TrainSet_ID,
                   use_images_instead_of_features=False,

                   verbose=True):

    start_time = time()

    predictions = []
    validate_y = []
    validate_ngrams = []

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

            sub_batch_sugg, sub_batch_sugg_num_pads = \
                data_loader.get_ngrams_by_idxes(range(from_idx, to_idx), dataset_split=dataset_split)
            max_num_ngrams = sub_batch_sugg.size(1)
            for i in range(len(sub_batch_sugg)):
                if sub_batch_sugg_num_pads[i] != 0:
                    actual_num_ngrams = max_num_ngrams - sub_batch_sugg_num_pads[i]
                    validate_ngrams.append(sub_batch_sugg[i, :actual_num_ngrams].tolist())
                else:
                    validate_ngrams.append(sub_batch_sugg[i].tolist())

            beam_search_kwargs = {'beam_size': beam_size,
                                  'beam_max_seq_len': max_seq_len,
                                  'sample_or_max': 'max',
                                  'how_many_outputs': 1,
                                  'sos_idx': sos_idx,
                                  'eos_idx': eos_idx}

            FINAL_PRED, _ = ddp_model(enc_x=sub_batch_x,
                    enc_x_num_pads=sub_batch_x_num_pads,
                    mode='beam_search', **beam_search_kwargs)
            list_pred = [pred[0] for pred in FINAL_PRED]

            predictions += list_pred

            del sub_batch_x, sub_batch_x_num_pads

    ddp_model.train()

    if rank == 0 and verbose:
        assert( len(predictions) == len(validate_ngrams) ), "Should have same num of entries"
        recall_mean = 0
        precision_mean = 0
        f1_mean = 0
        for pred, gt_ngrams, i in zip(predictions, validate_ngrams, range(len(predictions))):
            single_recall = 0
            for ngram_idx in gt_ngrams:
                for pred_ngram in pred:
                    if pred_ngram == ngram_idx:
                        single_recall += 1
                        break  # just count once! this is why we need a break

            single_recall = (single_recall / len(gt_ngrams))
            recall_mean += single_recall

            single_precision = 0
            for pred_ngram in pred:
                if pred_ngram in gt_ngrams:
                    single_precision += 1

            single_precision = (single_precision / len(pred))
            precision_mean += single_precision

            if (single_precision + single_recall) != 0:
                f1 = 2 * (single_precision * single_recall) / \
                     (single_precision + single_recall)
            else:
                f1 = 0
            f1_mean += f1

            #if i < 2:
            #    list_pred_tokens = [[ y_idx2word_list[idx] for idx in n_gram ] for n_gram in pred]
            #    list_gt_tokens = [[ y_idx2word_list[idx] for idx in n_gram ] for n_gram in gt_ngrams]
            #    print("Example PRED: " + str(list_pred_tokens), end=" @@@ ")
            #    print("s_REFS: " + str(list_gt_tokens), end=" @@@ ")
            #    print("s_recall: " + str(round(single_recall, 4)), end=" / ")
            #    print("s_precision: " + str(round(single_precision, 4)), end=" / ")
            #    print("f1: " + str(round(f1, 4)))

        print("[EVALUATION SUGGESTIONS] ::: Recall avg: " + str(round(recall_mean / len(predictions), 4)), end=" ")
        print("::: Precision avg: " + str(round(precision_mean / len(predictions), 4)), end=" ")
        print("::: F1 avg: " + str(round(f1_mean / len(predictions), 4)))


    if rank == 0:
        return None, None

    return None, None


def evaluate_unique_words_on_set(ddp_model,
          caption_idx2word_list,
          caption_word2idx_dict,
          ternary_word2idx_dict,
          sos_idx, eos_idx,
          num_samples,
          data_loader,
          dataset_split,
          eval_max_len,
          rank, ddp_sync_port,
          parallel_batches=16,
          beam_sizes=[1],
          stanford_model_path='./eval/get_stanford_models.sh',
          # stanford_model_path="/home/jchu/PASSAGGIO_WINDOWS/pap_12_Unet_Static_Expansion_CVPR/workspace_unet_sexp/eval/get_stanford_models.sh",
          use_images_instead_of_features=False,
          get_predictions=False):

    with torch.no_grad():
        ddp_model.eval()

        pred_dict, gts_dict = evaluate_unique_words(ddp_model,
             y_idx2word_list=caption_idx2word_list,
             y_word2idx_dict=caption_word2idx_dict,
             ternary_word2idx_dict=ternary_word2idx_dict,
             beam_size=3, max_seq_len=eval_max_len,
             sos_idx=sos_idx, eos_idx=eos_idx,
             rank=rank,
             ddp_sync_port=ddp_sync_port,
             parallel_batches=parallel_batches,
             indexes=list(range(num_samples)),
             data_loader=data_loader,
             dataset_split=dataset_split,
             use_images_instead_of_features=use_images_instead_of_features,
             verbose=True)

        if rank == 0 and get_predictions:
            return pred_dict, gts_dict

    return None, None


#########################################


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

    from models.suggestion_module.sst_sugg_module import SST_Sugg_Module
    model = SST_Sugg_Module(
        k_gram=test_args.selected_n_gram,
        d_model=model_args.model_dim,
        ff=model_args.model_dim * 4, num_heads=8,
        num_layers=model_args.N_enc, drop_args=model_args.drop_args,
        output_word2idx=coco_dataset.caption_word2idx_dict,
        output_idx2word=coco_dataset.caption_idx2word_list,
        max_seq_len=model_max_len, img_feature_dim=1536,
        num_diffusion_steps=20, rank=rank)

    print("Suggestion Module's class: " + str(model.__class__))
    model.to(rank)

    image_size = 384
    data_loader = CocoDataLoader(coco_dataset=coco_dataset,
                                 batch_size=1,
                                 num_procs=world_size,
                                 array_of_init_seeds=array_of_init_seeds,
                                 dataloader_mode='caption_wise',
                                 resize_image_size=image_size,
                                 rank=rank,
                                 verbose=True)
    ema = ModelEma(model, 0.999)

    checkpoint = torch.load(path_args.save_path)
    ema.load_state_dict(checkpoint['ema'])
    print("Loaded checkpoint: " + str(path_args.save_path))

    print("Evaluation on Validation Set")
    evaluate_unique_words_on_set(ema.module,
                                 coco_dataset.caption_idx2word_list,
                                 coco_dataset.caption_word2idx_dict,
                                 None,
                                 coco_dataset.get_sos_token_idx(), coco_dataset.get_eos_token_idx(),
                                 coco_dataset.val_num_images, data_loader,
                                 CocoDatasetKarpathy.ValidationSet_ID, model_max_len,
                                 rank, test_args.ddp_sync_port,
                                 parallel_batches=test_args.eval_parallel_batch_size,
                                 use_images_instead_of_features=False,
                                 beam_sizes=test_args.eval_beam_sizes)

    print("Evaluation on Test Set")
    evaluate_unique_words_on_set(ema.module,
                                 coco_dataset.caption_idx2word_list,
                                 coco_dataset.caption_word2idx_dict,
                                 None,
                                 coco_dataset.get_sos_token_idx(), coco_dataset.get_eos_token_idx(),
                                 coco_dataset.test_num_images, data_loader,
                                 CocoDatasetKarpathy.TestSet_ID, model_max_len,
                                 rank, test_args.ddp_sync_port,
                                 parallel_batches=test_args.eval_parallel_batch_size,
                                 use_images_instead_of_features=False,
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
    parser.add_argument('--save_path', type=str, default='./github_ignore_material/saves_suggestion_module/',
                        help='Checkpoint folder.')

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
    args = parser.parse_args()
    args.ddp_sync_port = str(args.ddp_sync_port)

    drop_args = Namespace(enc=args.enc_drop,
                          dec=args.dec_drop,
                          enc_input=args.enc_input_drop,
                          dec_input=args.dec_input_drop,
                          other=args.drop_other)

    model_args = Namespace(model_dim=args.model_dim,
                           N_enc=args.N_enc,
                           N_dec=args.N_dec,
                           drop_args=drop_args)

    path_args = Namespace(save_path=args.save_path,
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