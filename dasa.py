import numpy as np
import torch
import torch.nn as nn
import copy
from torchvision import transforms
import torch.nn.functional as F
import random

class AlternatingAttacker():
    def __init__(self, model, img_attacker, txt_attacker, steps=10):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.steps = steps
        if hasattr(self.img_attacker, 'model'):
            self.img_attacker.model = model

    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, image_steps=10, **kwargs):
        imgs = imgs.to(device)
        adv_imgs = imgs.clone()
        adv_txts = copy.deepcopy(txts)
        
        for step in range(self.steps):
            pass
            
        raise NotImplementedError()
        return adv_imgs, adv_txts

class ImageAttacker1():
    def __init__(self, normalization, eps=8 / 255, step_size=2 / 255,
                 mu=1, local_weight=0.5, prob_di=0.7):
        self.normalization = normalization
        self.eps = eps
        self.step_size = step_size
        self.mu = mu
        self.local_weight = local_weight
        self.prob_di = prob_di

    def spatial_warping(self, x, distortion_scale=0.3):    
        return transforms.RandomPerspective(distortion_scale=distortion_scale, p=1.0)(x)

        #### Core functions

class TextAttacker1():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls


    def _tokenize(self, text):
        words = text.split(' ')
        sub_words, keys, index = [], [], 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            if not sub:
                keys.append([index, index])
            else:
                sub_words += sub
                keys.append([index, index + len(sub)])
                index += len(sub)
        return words, sub_words, keys



filter_words = set(['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·'])

def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    for (i, j) in zip(substitutes[0], substitutes_score[0]):
        if threshold != 0 and j < threshold: break
        words.append(tokenizer._convert_id_to_token(int(i)))
    return words
