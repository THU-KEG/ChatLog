from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from typing import Callable, List, Tuple
import os
import pickle
from nltk.data import load as nltk_load
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda"
from_local = True
local_qa_en = "/data/tsq/CK/ckpt/chatgpt-qa-detector-roberta"
local_single_en = "/data/tsq/CK/ckpt/chatgpt-detector-single"
local_ling = "/data/tsq/CK/ckpt/ling"
local_zh = "/data/tsq/CK/ckpt/chatgpt-qa-detector-roberta-chinese"

# default models
# TOKENIZER_EN, MODEL_EN, sent_cut_en = None, None, None
# LR_GLTR_EN, LR_PPL_EN = None, None
# tokenizer_qa_en, model_qa_en = None, None
# tokenizer_single_en, model_single_en = None, None
# model_zh = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-qa-detector-roberta").to(device)
classes = ["human", "ChatGPT"]

CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')


class Detector:
    def __init__(self):
        if from_local:
            # qa model
            self.tokenizer_qa_en = AutoTokenizer.from_pretrained(local_qa_en)
            self.model_qa_en = AutoModelForSequenceClassification.from_pretrained(local_qa_en).to(device)
            # single text
            self.tokenizer_single_en = AutoTokenizer.from_pretrained(local_single_en)
            self.model_single_en = AutoModelForSequenceClassification.from_pretrained(local_single_en).to(device)
            # ling
            self.NLTK = nltk_load(os.path.join(local_ling, 'english.pickle'))
            self.sent_cut_en = self.NLTK.tokenize
            self.LR_GLTR_EN, self.LR_PPL_EN, self.LR_GLTR_ZH, self.LR_PPL_ZH = [
                pickle.load(open(os.path.join(local_ling, f'{lang}-gpt2-{name}.pkl'), 'rb'))
                for lang, name in [('en', 'gltr'), ('en', 'ppl'), ('zh', 'gltr'), ('zh', 'ppl')]
            ]

            self.NAME_EN = 'gpt2'
            self.TOKENIZER_EN = GPT2Tokenizer.from_pretrained(self.NAME_EN)
            self.MODEL_EN = GPT2LMHeadModel.from_pretrained(self.NAME_EN)

        else:
            self.tokenizer_en = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-qa-detector-roberta")
            self.tokenizer_zh = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-qa-detector-roberta-chinese")
            self.model_en = AutoModelForSequenceClassification.from_pretrained(
                "Hello-SimpleAI/chatgpt-qa-detector-roberta").to(
                device)
            self.model_zh = AutoModelForSequenceClassification.from_pretrained(
                "Hello-SimpleAI/chatgpt-qa-detector-roberta-chinese").to(
                device)

    def get_classification(self, request_data):
        q = request_data["q"]
        a = request_data["a"]
        language = request_data["language"]
        prob1, label1 = self.classification(q, a, language, method="qa")
        prob2, label2 = self.classification(q, a, language, method="single")
        # [id_to_label[gltr_label], gltr_prob, id_to_label[ppl_label], ppl_prob]
        ling_lst = self.classification(q, a, language, method="ling")
        res = {
            'prob': [prob1, prob2, ling_lst[1], ling_lst[3]],
            'label': [label1, label2, ling_lst[0], ling_lst[2]],
            "method": ["qa", "single", "gltr", "ppl"]
        }
        return res

    def gpt2_features(
            self, text: str, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, sent_cut: Callable
    ) -> Tuple[List[int], List[float]]:
        # Tokenize
        input_max_length = tokenizer.model_max_length - 2
        token_ids, offsets = list(), list()
        sentences = sent_cut(text)
        for s in sentences:
            tokens = tokenizer.tokenize(s)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            difference = len(token_ids) + len(ids) - input_max_length
            if difference > 0:
                ids = ids[:-difference]
            offsets.append((len(token_ids), len(token_ids) + len(ids)))  # 左开右闭
            token_ids.extend(ids)
            if difference >= 0:
                break

        input_ids = torch.tensor([tokenizer.bos_token_id] + token_ids)
        logits = model(input_ids).logits
        # Shift so that n-1 predict n
        shift_logits = logits[:-1].contiguous()
        shift_target = input_ids[1:].contiguous()
        loss = CROSS_ENTROPY(shift_logits, shift_target)

        all_probs = torch.softmax(shift_logits, dim=-1)
        sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)  # stable=True
        expanded_tokens = shift_target.unsqueeze(-1).expand_as(sorted_ids)
        indices = torch.where(sorted_ids == expanded_tokens)
        rank = indices[-1]
        counter = [
            rank < 10,
            (rank >= 10) & (rank < 100),
            (rank >= 100) & (rank < 1000),
            rank >= 1000
        ]
        counter = [c.long().sum(-1).item() for c in counter]

        # compute different-level ppl
        text_ppl = loss.mean().exp().item()
        sent_ppl = list()
        for start, end in offsets:
            nll = loss[start: end].sum() / (end - start)
            sent_ppl.append(nll.exp().item())
        max_sent_ppl = max(sent_ppl)
        sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
        if len(sent_ppl) > 1:
            sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item()
        else:
            sent_ppl_std = 0

        mask = torch.tensor([1] * loss.size(0))
        step_ppl = loss.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
        max_step_ppl = step_ppl.max(dim=-1)[0].item()
        step_ppl_avg = step_ppl.sum(dim=-1).div(loss.size(0)).item()
        if step_ppl.size(0) > 1:
            step_ppl_std = step_ppl.std().item()
        else:
            step_ppl_std = 0
        ppls = [
            text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
            max_step_ppl, step_ppl_avg, step_ppl_std
        ]
        return counter, ppls  # type: ignore

    def lr_predict(
            self, f_gltr: List[int], f_ppl: List[float], lr_gltr: LogisticRegression, lr_ppl: LogisticRegression,
            id_to_label: List[str]
    ) -> List:
        x_gltr = np.asarray([f_gltr])
        gltr_label = lr_gltr.predict(x_gltr)[0]
        gltr_prob = lr_gltr.predict_proba(x_gltr)[0, gltr_label]
        x_ppl = np.asarray([f_ppl])
        ppl_label = lr_ppl.predict(x_ppl)[0]
        ppl_prob = lr_ppl.predict_proba(x_ppl)[0, ppl_label]
        return [id_to_label[gltr_label], gltr_prob * 100, id_to_label[ppl_label], ppl_prob * 100]

    def predict_en_ling(self, text: str) -> List:
        with torch.no_grad():
            feat = self.gpt2_features(text, self.TOKENIZER_EN, self.MODEL_EN, self.sent_cut_en)
        out = self.lr_predict(*feat, self.LR_GLTR_EN, self.LR_PPL_EN, ['human', 'ChatGPT'])
        return out

    def pipeline_hc3_qa(self, q, a, language="en"):
        batch = self.tokenizer_qa_en(q, a, return_tensors="pt").to(device)
        # print(batch)
        with torch.no_grad():
            outputs = self.model_qa_en(**batch)
        # print(outputs)
        logits = outputs[0]
        predictions = torch.softmax(logits, dim=1).tolist()[0]
        probabilities = [predictions[i] * 100 for i in range(2)]
        return probabilities

    def pipeline_hc3_single(self, q, a, language="en"):
        batch = self.tokenizer_single_en(a, return_tensors="pt").to(device)
        # print(batch)
        with torch.no_grad():
            outputs = self.model_single_en(**batch)
        # print(outputs)
        logits = outputs[0]
        predictions = torch.softmax(logits, dim=1).tolist()[0]
        probabilities = [predictions[i] * 100 for i in range(2)]
        return probabilities

    def classification(self, q, a, language, method="qa"):
        if method != "ling":
            prob, label = 0, "human"
            if method == "qa":
                probabilities = self.pipeline_hc3_qa(q, a, language)
            else:
                # single
                probabilities = self.pipeline_hc3_single(q, a, language)
            is_human_prob = probabilities[0]
            is_bot_prob = probabilities[1]
            if is_human_prob > is_bot_prob:
                prob = is_human_prob
            else:
                prob = is_bot_prob
                label = classes[1]
            return prob, label
        else:
            # ling
            return self.predict_en_ling(a)
