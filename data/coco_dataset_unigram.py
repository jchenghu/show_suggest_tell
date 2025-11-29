import json
from time import time
from utils import language_utils
from collections import defaultdict


import functools
print = functools.partial(print, flush=True)


class CocoDatasetKarpathy:

    TrainSet_ID = 1
    ValidationSet_ID = 2
    TestSet_ID = 3

    def __init__(self,
                 k_gram,
                 images_path,
                 coco_annotations_path,
                 precalc_features_hdf5_filepath,
                 preproc_images_hdf5_filepath=None,
                 limited_num_train_images=None,
                 limited_num_val_images=None,
                 limited_num_test_images=None,
                 dict_min_occurrences=5,
                 verbose=True
                 ):
        super(CocoDatasetKarpathy, self).__init__()

        self.k_gram = k_gram
        self.use_images_instead_of_features = False
        if precalc_features_hdf5_filepath is None or precalc_features_hdf5_filepath == 'None' or \
                precalc_features_hdf5_filepath == 'none' or precalc_features_hdf5_filepath == '':
            self.use_images_instead_of_features = True
            print("Warning: since no hdf5 path is provided using images instead of pre-calculated features.")
            print("Features path: " + str(precalc_features_hdf5_filepath))

            self.preproc_images_hdf5_filepath = None
            if preproc_images_hdf5_filepath is not None:
                print("Preprocessed hdf5 file path not None: " + str(preproc_images_hdf5_filepath))
                print("Using preprocessed hdf5 file instead.")
                self.preproc_images_hdf5_filepath = preproc_images_hdf5_filepath

        else:
            self.precalc_features_hdf5_filepath = precalc_features_hdf5_filepath
            print("Features path: " + str(self.precalc_features_hdf5_filepath))
            print("Features path provided, images are provided in form of features.")

        if images_path is None:
            self.images_path = ""
        else:
            self.images_path = images_path

        self.karpathy_train_dict = dict()
        self.karpathy_val_dict = dict()
        self.karpathy_test_dict = dict()

        with open(coco_annotations_path, 'r') as f:
            json_file = json.load(f)['images']

        if verbose:
            print("Initializing dataset... ", end=" ")
        for json_item in json_file:
            new_item = dict()

            new_item['img_path'] = self.images_path + json_item['filepath'] + '/img/' + json_item['filename']

            new_item_captions = [item['raw'] for item in json_item['sentences']]
            new_item['img_id'] = json_item['cocoid']
            new_item['captions'] = new_item_captions

            if json_item['split'] == 'train' or json_item['split'] == 'restval':
                self.karpathy_train_dict[json_item['cocoid']] = new_item
            elif json_item['split'] == 'test':
                self.karpathy_test_dict[json_item['cocoid']] = new_item
            elif json_item['split'] == 'val':
                self.karpathy_val_dict[json_item['cocoid']] = new_item

        self.karpathy_train_list = []
        self.karpathy_val_list = []
        self.karpathy_test_list = []
        for key in self.karpathy_train_dict.keys():
            self.karpathy_train_list.append(self.karpathy_train_dict[key])
        for key in self.karpathy_val_dict.keys():
            self.karpathy_val_list.append(self.karpathy_val_dict[key])
        for key in self.karpathy_test_dict.keys():
            self.karpathy_test_list.append(self.karpathy_test_dict[key])

        self.train_num_images = len(self.karpathy_train_list)
        self.val_num_images = len(self.karpathy_val_list)
        self.test_num_images = len(self.karpathy_test_list)

        if limited_num_train_images is not None:
            self.karpathy_train_list = self.karpathy_train_list[:limited_num_train_images]
            self.train_num_images = limited_num_train_images
        if limited_num_val_images is not None:
            self.karpathy_val_list = self.karpathy_val_list[:limited_num_val_images]
            self.val_num_images = limited_num_val_images
        if limited_num_test_images is not None:
            self.karpathy_test_list = self.karpathy_test_list[:limited_num_test_images]
            self.test_num_images = limited_num_test_images

        if verbose:
            print("Num train images: " + str(self.train_num_images))
            print("Num val images: " + str(self.val_num_images))
            print("Num test images: " + str(self.test_num_images))

        tokenized_captions_list = []
        for i in range(self.train_num_images):
            for caption in self.karpathy_train_list[i]['captions']:
                tmp = language_utils.lowercase_and_clean_trailing_spaces([caption])
                tmp = language_utils.add_space_between_non_alphanumeric_symbols(tmp)
                tmp = language_utils.remove_punctuations(tmp)
                tokenized_caption = ['SOS'] + language_utils.tokenize(tmp)[0] + ['EOS']
                tokenized_captions_list.append(tokenized_caption)

        counter_dict = dict()
        for i in range(len(tokenized_captions_list)):
            for word in tokenized_captions_list[i]:
                if word not in counter_dict:
                    counter_dict[word] = 1
                else:
                    counter_dict[word] += 1

        less_than_min_occurrences_set = set()
        for k, v in counter_dict.items():
            if v < dict_min_occurrences:
                less_than_min_occurrences_set.add(k)
        if verbose:
            print("tot tokens " + str(len(counter_dict)) +
                  " less than " + str(dict_min_occurrences) + ": " + str(len(less_than_min_occurrences_set)) +
                  " remaining: " + str(len(counter_dict) - len(less_than_min_occurrences_set)))

        self.num_caption_vocab = 4
        self.max_seq_len = 0
        discovered_words = ['PAD', 'SOS', 'EOS', 'UNK', 'MASK']
        for i in range(len(tokenized_captions_list)):
            caption = tokenized_captions_list[i]
            if len(caption) > self.max_seq_len:
                self.max_seq_len = len(caption)
            for word in caption:
                if (word not in discovered_words) and (not word in less_than_min_occurrences_set):
                    discovered_words.append(word)
                    self.num_caption_vocab += 1

        discovered_words.sort()
        self.caption_word2idx_dict = dict()
        self.caption_idx2word_list = []
        for i in range(len(discovered_words)):
            self.caption_word2idx_dict[discovered_words[i]] = i
            self.caption_idx2word_list.append(discovered_words[i])
        if verbose:
            print("There are " + str(self.num_caption_vocab) + " vocabs in dict")

        ###########################################################################################
        ###########################################################################################

        print("Selected N_GRAM atm: " + str(k_gram))
        assert (k_gram <= 4), "Implementation for now supports up to 4-grams"


        all_raw_sentences = self.get_all_images_captions(dataset_split=CocoDatasetKarpathy.TrainSet_ID)
        # Borrowed dal dataloader...
        processed_corpus = []
        for sentences in all_raw_sentences:
            processed_corpus.append([' '.join(self.preprocess(sent)) for sent in sentences])

        from eval.cider.reinforce_cider import ReinforceCiderScorer

        self.cider_scorer = ReinforceCiderScorer(processed_corpus, n=4, sigma=1.0)

        count_1_gram = 0
        count_2_gram = 0
        count_3_gram = 0
        count_4_gram = 0
        self.vocab_k_gram2idx = dict()
        self.vocab_k_idx2gram = []
        for key in self.cider_scorer.document_frequency.keys():
            if len(key) == 1:
                if k_gram == 1:
                    self.vocab_k_gram2idx[self.convert_ngram2string(key)] = count_1_gram
                    self.vocab_k_idx2gram.append(self.convert_ngram2string(key))
                count_1_gram += 1
            elif len(key) == 2:
                if k_gram == 2:
                    self.vocab_k_gram2idx[self.convert_ngram2string(key)] = count_2_gram
                    self.vocab_k_idx2gram.append(self.convert_ngram2string(key))
                count_2_gram += 1
            elif len(key) == 3:
                if k_gram == 3:
                    self.vocab_k_gram2idx[self.convert_ngram2string(key)] = count_3_gram
                    self.vocab_k_idx2gram.append(self.convert_ngram2string(key))
                count_3_gram += 1
            elif len(key) == 4:
                if k_gram == 4:
                    self.vocab_k_gram2idx[self.convert_ngram2string(key)] = count_4_gram
                    self.vocab_k_idx2gram.append(self.convert_ngram2string(key))
                count_4_gram += 1
        print("[CHECK] Count 1-gram: " + str(count_1_gram))
        print("[CHECK] Count 2-gram: " + str(count_2_gram))
        print("[CHECK] Count 3-gram: " + str(count_3_gram))
        print("[CHECK] Count 4-gram: " + str(count_4_gram))

        def precook_specific(s, n):
            """
            Takes a string as input and returns an object that can be given to
            either cook_refs or cook_test. This is optional: cook_refs and cook_test
            can take string arguments as well.
            :param s: string : sentence to be converted into ngrams
            :param n: int    : number of ngrams for which representation is calculated
            :return: term frequency vector for occuring ngrams
            """
            words = s.split()
            counts = defaultdict(int)
            #for k in range(1, n + 1):
            k = n
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i:i + k])
                counts[ngram] += 1
            return counts

        print("Creating the ngram set for each sentence")
        for i in range(self.train_num_images):
            caption_ngram_set_str = []
            imgwise_ngram_set_str = []
            for caption in self.karpathy_train_list[i]['captions']:
                processed_caption = ' '.join(self.preprocess(caption))
                found_ngrams = precook_specific(processed_caption, self.k_gram)
                caption_ngram_set_str.append([self.convert_ngram2string(ngram) \
                    for ngram in found_ngrams.keys()])
                for ngram in found_ngrams.keys():
                    if self.convert_ngram2string(ngram) not in imgwise_ngram_set_str:
                        imgwise_ngram_set_str.append(self.convert_ngram2string(ngram))

            self.karpathy_train_list[i]['ngrams_str'] = caption_ngram_set_str
            self.karpathy_train_list[i]['ngrams_idx'] = \
                [[self.vocab_k_gram2idx[ngram_str] for ngram_str in ngram_str_sentence] for ngram_str_sentence in caption_ngram_set_str]
            self.karpathy_train_list[i]['ngrams_imgwise_str'] = imgwise_ngram_set_str
            self.karpathy_train_list[i]['ngrams_imgwise_idx'] = \
                [self.vocab_k_gram2idx[ngram_str] for ngram_str in imgwise_ngram_set_str]

        print("How gram2idx: " + str(len(self.vocab_k_gram2idx)))
        print("How idx2gram: " + str(len(self.vocab_k_idx2gram)))

        for i in range(self.val_num_images):
            caption_ngram_set_str = []
            imgwise_ngram_set_str = []
            for caption in self.karpathy_val_list[i]['captions']:
                processed_caption = ' '.join(self.preprocess(caption))
                found_ngrams = precook_specific(processed_caption, self.k_gram)
                caption_ngram_set_str.append([self.convert_ngram2string(ngram) \
                                              for ngram in found_ngrams.keys()
                                              if self.convert_ngram2string(ngram) in self.vocab_k_gram2idx.keys()
                                              ])
                for ngram in found_ngrams.keys():
                    if self.convert_ngram2string(ngram) in self.vocab_k_gram2idx.keys():
                        if self.convert_ngram2string(ngram) not in imgwise_ngram_set_str:
                            imgwise_ngram_set_str.append(self.convert_ngram2string(ngram))

            self.karpathy_val_list[i]['ngrams_str'] = caption_ngram_set_str
            self.karpathy_val_list[i]['ngrams_idx'] = \
                [[self.vocab_k_gram2idx[ngram_str] for ngram_str in ngram_str_sentence] for ngram_str_sentence in
                 caption_ngram_set_str]
            self.karpathy_val_list[i]['ngrams_imgwise_str'] = imgwise_ngram_set_str
            self.karpathy_val_list[i]['ngrams_imgwise_idx'] = \
                [self.vocab_k_gram2idx[ngram_str] for ngram_str in imgwise_ngram_set_str]

        for i in range(self.test_num_images):
            caption_ngram_set_str = []
            imgwise_ngram_set_str = []
            for caption in self.karpathy_test_list[i]['captions']:
                processed_caption = ' '.join(self.preprocess(caption))
                found_ngrams = precook_specific(processed_caption, self.k_gram)
                caption_ngram_set_str.append([self.convert_ngram2string(ngram) \
                                              for ngram in found_ngrams.keys() \
                                              if self.convert_ngram2string(ngram) in self.vocab_k_gram2idx.keys()])
                for ngram in found_ngrams.keys():
                    if self.convert_ngram2string(ngram) in self.vocab_k_gram2idx.keys():
                        if self.convert_ngram2string(ngram) not in imgwise_ngram_set_str:
                            imgwise_ngram_set_str.append(self.convert_ngram2string(ngram))

            self.karpathy_test_list[i]['ngrams_str'] = caption_ngram_set_str
            self.karpathy_test_list[i]['ngrams_idx'] = \
                [[self.vocab_k_gram2idx[ngram_str] for ngram_str in ngram_str_sentence] for ngram_str_sentence in
                 caption_ngram_set_str]
            self.karpathy_test_list[i]['ngrams_imgwise_str'] = imgwise_ngram_set_str
            self.karpathy_test_list[i]['ngrams_imgwise_idx'] = \
                [self.vocab_k_gram2idx[ngram_str] for ngram_str in imgwise_ngram_set_str]

        self.fast_vocab_check_ngram = set([key for key in self.vocab_k_gram2idx.keys()])


    def check_ngram_in_vocab(self, ngram):
        words_list = [self.caption_idx2word_list[idx] for idx in ngram]
        ngram_string = self.convert_ngram2string(words_list)
        return ngram_string in self.fast_vocab_check_ngram

    def convert_ngram2string(self, ngram):
        s = ''
        for piece in ngram:
            s += '_' + piece
        return s


    def convert_string2listngram(self, s):
        # e.g. '_agas_bga'.split('_')  -> ['', 'agas', 'bga']
        # discards the last piece
        return s.split('_')[1:]

    def preprocess(self, caption):
        caption = language_utils.lowercase_and_clean_trailing_spaces([caption])
        caption = language_utils.add_space_between_non_alphanumeric_symbols(caption)
        caption = language_utils.remove_punctuations(caption)
        caption = [self.get_sos_token_str()] + language_utils.tokenize(caption)[0] + [
            self.get_eos_token_str()]
        preprocessed_tokenized_caption = []
        for word in caption:
            if word not in self.caption_word2idx_dict.keys():
                preprocessed_tokenized_caption.append(self.get_unk_token_str())
            else:
                preprocessed_tokenized_caption.append(word)
        return preprocessed_tokenized_caption

    def get_image_path(self, img_idx, dataset_split):

        if dataset_split == CocoDatasetKarpathy.TestSet_ID:
            img_path = self.karpathy_test_list[img_idx]['img_path']
            img_id = self.karpathy_test_list[img_idx]['img_id']
        elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
            img_path = self.karpathy_val_list[img_idx]['img_path']
            img_id = self.karpathy_val_list[img_idx]['img_id']
        else:
            img_path = self.karpathy_train_list[img_idx]['img_path']
            img_id = self.karpathy_train_list[img_idx]['img_id']

        return img_path, img_id

    def get_all_images_captions(self, dataset_split):
        all_image_references = []

        if dataset_split == CocoDatasetKarpathy.TestSet_ID:
            dataset = self.karpathy_test_list
        elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
            dataset = self.karpathy_val_list
        else:
            dataset = self.karpathy_train_list

        for img_idx in range(len(dataset)):
            all_image_references.append(dataset[img_idx]['captions'])
        return all_image_references

    def get_eos_token_idx(self):
        return self.caption_word2idx_dict['EOS']

    def get_sos_token_idx(self):
        return self.caption_word2idx_dict['SOS']

    def get_pad_token_idx(self):
        return self.caption_word2idx_dict['PAD']

    def get_unk_token_idx(self):
        return self.caption_word2idx_dict['UNK']

    def get_mask_token_idx(self):
        return self.caption_word2idx_dict['MASK']

    def get_eos_token_str(self):
        return 'EOS'

    def get_sos_token_str(self):
        return 'SOS'

    def get_pad_token_str(self):
        return 'PAD'

    def get_unk_token_str(self):
        return 'UNK'

    def get_mask_token_str(self):
        return 'MASK'
