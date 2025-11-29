# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

# ReinforceCIDEr is an alternative implementation of CIDEr where the corpus is initialized in the constructor
# so it doesn't need to be processed again every time we need to compute the cider score
# in the Self Critical Learning Process. --- Jia Cheng

from eval.cider.reinforce_cider_scorer import ReinforceCiderScorer
import pdb


class ReinforceCider:

    # The batch_ref_sentences will be a small sample of the original corpus, note however that there's no need of
    # correspondence of img_ids between img_ids in the corpus and the ones in the batch_ref_sentences, the img_ids
    # consistency is required between batch_ref_sentences and batch_test_sentences only.
    def __init__(self,  corpus, n=4, sigma=6.0):
        '''
        Corpus represents the collection of reference sentences for each image, this must be a dictionary with image
        ids as keys and a list of sentences as value.

        :param corpus: a dictionary with
        :param n: number of n-grams
        :param sigma: length penalty coefficient
        '''
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        self.cider_scorer = ReinforceCiderScorer(corpus, n=self._n, sigma=self._sigma)

        """
        print("Creato CIDEr scorer exito: ")
        print(self.cider_scorer.document_frequency)
        print("\n\n\n\n")
        print("Which keys: " + str(self.cider_scorer.document_frequency.keys()))
        print("\n\n\n\n")
        print("How many keys: " + str(len(self.cider_scorer.document_frequency.keys())))
        count_1_gram = 0
        count_2_gram = 0
        count_3_gram = 0
        count_4_gram = 0
        for key in self.cider_scorer.document_frequency.keys():
            if len(key) == 1:
                count_1_gram += 1
            elif len(key) == 2:
                count_2_gram += 1
            elif len(key) == 3:
                count_3_gram += 1
            elif len(key) == 4:
                count_4_gram += 1
        print("Count 1-gram: " + str(count_1_gram))
        print("Count 2-gram: " + str(count_2_gram))
        print("Count 3-gram: " + str(count_3_gram))
        print("Count 4-gram: " + str(count_4_gram))

        count_1_gram = 0
        count_2_gram = 0
        count_3_gram = 0
        count_4_gram = 0
        for key in self.cider_scorer.document_frequency.keys():
            if len(key) == 1 and self.cider_scorer.document_frequency[key] >= 5:
                count_1_gram += 1
            elif len(key) == 2 and self.cider_scorer.document_frequency[key] >= 5:
                count_2_gram += 1
            elif len(key) == 3 and self.cider_scorer.document_frequency[key] >= 5:
                count_3_gram += 1
            elif len(key) == 4 and self.cider_scorer.document_frequency[key] >= 5:
                count_4_gram += 1
        print("Filter 5 Count 1-gram: " + str(count_1_gram))
        print("Filter 5 Count 2-gram: " + str(count_2_gram))
        print("Filter 5 Count 3-gram: " + str(count_3_gram))
        print("Filter 5 Count 4-gram: " + str(count_4_gram))
        exit(-1)
        """

    def compute_score(self, hypo, refs):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """

        # assert(hypo.keys() == refs.keys())

        (score, scores) = self.cider_scorer.compute_score(refs, hypo)

        return score, scores

    def method(self):
        return "Reinforce CIDEr"