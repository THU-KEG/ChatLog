from tqdm import tqdm
from pprint import pprint
from paddlenlp import Taskflow
import cogie
from config.conf import CONFIG
import requests
from lingfeat import extractor
from data.save_data import DBLoader
from model.detection import Detector

lang2uie_schema = {
    "en": ['Person', 'Organization', 'Location', 'Work', 'Time',
           'Opinion', 'Sentiment classification [negative, neutral, positive]'],
    "zh": ["人名", "机构名", "地名", "作品名", "时间",
           '观点词', '情感倾向[正向, 中立, 负向]'],
}

cogie_schema = ["words", "ner", "re", "fn", "argument"]

lang2uie = {
    "en": 'uie-base-en',
    "zh": 'uie-base'
}


class Extractor:
    def __init__(self, args):
        self.args = args
        if args.language == "zh":
            self.ner = Taskflow("ner")
        # schema = ['Person', 'Organization']
        # schema = 'Sentiment classification [negative, positive]'
        # schema = 'Opinion'
        self.uie_schema = lang2uie_schema[args.language]
        if args.feature_type == "knowledge" or args.feature_type == "all":
            if args.knowledge_extractor == "uie" or args.knowledge_extractor == "all":
                self.ie = Taskflow('information_extraction',
                                   schema=self.uie_schema,
                                   model=lang2uie[args.language])
            if args.knowledge_extractor == "cogie" or args.knowledge_extractor == "all":
                self.cog_tokenize_toolkit = cogie.TokenizeToolkit(task='ws', language='english', corpus=None)
                self.cog_ner_toolkit = cogie.NerToolkit(task='ner', language='english', corpus='trex')
                self.cog_re_toolkit = cogie.ReToolkit(task='re', language='english', corpus='trex')
                # self.cog_el_toolkit = cogie.ElToolkit(task='el', language='english', corpus='wiki')
                self.cog_fn_toolkit = cogie.FnToolkit(task='fn', language='english', corpus=None)
                self.cog_argument_toolkit = cogie.ArgumentToolkit(task='fn', language='english', corpus='argument')
        if args.feature_type == "classify" or args.feature_type == "all":
            self.detector = Detector()

        self.ling_feats = []
        self.loader = DBLoader(args)
        self.questions = self.loader.load_questions()

    def get_opinion(self, sentence):
        schema = 'Sentiment classification [negative, positive]'
        self.ie.set_schema(schema)
        op_result = self.ie(sentence)
        pprint(op_result)
        return op_result

    def get_ner(self, sentences):
        entities = self.ner(sentences)
        return entities

    def extract_once_cogie(self, words):
        try:
            ner_result = self.cog_ner_toolkit.run(words)
        except Exception:
            ner_result = []
        # relation extraction
        try:
            re_result = self.cog_re_toolkit.run(words, ner_result)
        except Exception:
            re_result = []
        # entity linking
        # el_result = self.cog_el_toolkit.run(ner_result)
        # frame identification
        try:
            fn_result = self.cog_fn_toolkit.run(words)
        except Exception:
            fn_result = []
        # argument identification
        try:
            argument_result = self.cog_argument_toolkit.run(words, fn_result)
        except Exception:
            argument_result = []
        # final ["words", "ner", "re", "fn", "argument"]
        final_dict = {
            "words": words,
            "ner": ner_result,
            "re": re_result,
            "fn": fn_result,
            "argument": argument_result
        }
        return final_dict

    def extract_cogie_feature(self, text):
        # tokenize sentence into words
        words = self.cog_tokenize_toolkit.run(text)
        len_words = len(words)
        batch_len = 128
        num_batch = len_words // batch_len
        # named entity recognition
        # print(f"len of words {len(words)}")
        final_dict = {
            "words": [],
            "ner": [],
            "re": [],
            "fn": [],
            "argument": []
        }
        for i in range(num_batch + 1):
            start = i * batch_len
            end = (i + 1) * batch_len
            _words = words[start: end]
            res_dict = self.extract_once_cogie(_words)
            for k, lst in res_dict.items():
                final_dict[k].extend(lst)
        return final_dict

    def extract_by_cogie(self, texts):
        """
        :param texts: [str]
        :return: Dict, key: feature name, value: [str] all properties of cogie knowledge
        """
        total_len = len(texts)
        # init
        aggregate_features = {}
        for k in cogie_schema:
            aggregate_features[k] = []
        # make mini batch
        for i in tqdm(range(total_len), total=total_len):
            sent = texts[i]
            _feature = self.extract_cogie_feature(sent)
            for k in cogie_schema:
                aggregate_features[k].append(_feature[k])

        return aggregate_features

    def extract_by_uie(self, texts):
        """
        :param texts: [str]
        :return: Dict, key: feature name, value: [str] all properties of uie knowledge
        """
        total_len = len(texts)
        # init
        aggregate_features = {}
        for k in self.uie_schema:
            aggregate_features[k] = []
        # make mini batch
        for i in tqdm(range(total_len), total=total_len):
            sent_batch = [texts[i]]
            _feature = self.ie(sent_batch)[0]
            for k in self.uie_schema:
                feature_results = []
                try:
                    feature_results.extend(_feature[k])
                except KeyError:
                    # default feature_result is empty list
                    pass
                aggregate_features[k].append(feature_results)

        return aggregate_features

    def extract_knowledge_features(self, texts):
        """
        :param texts: [str]
        :return: Dict, key: feature name, value: [str] all properties of knowledge
        """
        aggregate_features = {}
        if self.args.knowledge_extractor == "uie" or self.args.knowledge_extractor == "all":
            uie_features = self.extract_by_uie(texts)
            aggregate_features.update(uie_features)

        if self.args.knowledge_extractor == "cogie" or self.args.knowledge_extractor == "all":
            cogie_features = self.extract_by_cogie(texts)
            aggregate_features.update(cogie_features)

        return aggregate_features

    def extract_linguistic_features(self, texts):
        """
        :param texts: [str]
        :return: Dict, key: feature name, value: [Float], each is a linguistic result
        """
        total_len = len(texts)
        linguistic_features = []
        aggregate_features = {}
        # make batch
        for i in tqdm(range(0, total_len), total=total_len):
            sent = texts[i]
            # get linguistic feature
            res = {}
            if self.args.language == 'en':
                res = self.get_linguistic(sent)
            linguistic_features.append(res)
            for k, value in res.items():
                try:
                    aggregate_features[k].append(value)
                except KeyError:
                    self.ling_feats.append(k)
                    aggregate_features[k] = [value]

        return aggregate_features

    def extract_classify_features(self, texts):
        """
        :param texts: [str]
        :return: Dict, key: feature name, value: [Float], each is a classify result
        """
        total_len = len(texts)
        aggregate_features = {"qa": [], "single": [], "gltr": [], "ppl": []}
        if self.args.extract_source == "human_answers":
            questions = []
            for q in self.questions:
                questions.extend([q] * 3)
        else:
            questions = self.questions
        assert len(texts) == len(questions)
        # make mini batch
        for i in tqdm(range(0, total_len), total=total_len):
            sent = texts[i]
            # get classify feature
            question = questions[i]
            _features, methods = self.get_HC3_classify_bot_feature(question, sent, self.args.language)
            for j, method in enumerate(methods):
                aggregate_features[method].append(_features[j])

        return aggregate_features

    def get_linguistic(self, text):
        """
        Preprocess text options (all boolean):
            - short (default False): include short words of < 3 letters
            - see_token (default False): return token list
            - see_sent_token (default False): return tokens in sentences
        output:
            - n_token
            - n_sent
            - token_list (optional)
            - sent_token_list (optional)
        """
        res = {}
        LingFeat = extractor.pass_text(text)
        LingFeat.preprocess()
        # or
        # print(LingFeat.preprocess())

        """Extract feature search method returns a dictionary of the corresponding features"""
        # Advanced Semantic (AdSem) Features
        if 'AdSem' in self.args.linguistic_features:
            WoKF = LingFeat.WoKF_()  # Wikipedia Knowledge Features
            WBKF = LingFeat.WBKF_()  # WeeBit Corpus Knowledge Features
            OSKF = LingFeat.OSKF_()  # OneStopEng Corpus Knowledge Features
            res.update(WoKF)
            res.update(WBKF)
            res.update(OSKF)

        # Discourse (Disco) Features
        if 'Disco' in self.args.linguistic_features:
            EnDF = LingFeat.EnDF_()  # Entity Density Features
            EnGF = LingFeat.EnGF_()  # Entity Grid Features
            res.update(EnDF)
            res.update(EnGF)

        # Syntactic (Synta) Features
        if 'Synta' in self.args.linguistic_features:
            PhrF = LingFeat.PhrF_()  # Noun/Verb/Adj/Adv/... Phrasal Features
            TrSF = LingFeat.TrSF_()  # (Parse) Tree Structural Features
            POSF = LingFeat.POSF_()  # Noun/Verb/Adj/Adv/... Part-of-Speech Features
            res.update(PhrF)
            res.update(TrSF)
            res.update(POSF)

        # Lexico Semantic (LxSem) Features
        if 'LxSem' in self.args.linguistic_features:
            TTRF = LingFeat.TTRF_()  # Type Token Ratio Features
            VarF = LingFeat.VarF_()  # Noun/Verb/Adj/Adv Variation Features
            PsyF = LingFeat.PsyF_()  # Psycholinguistic Difficulty of Words (AoA Kuperman)
            WoLF = LingFeat.WorF_()  # Word Familiarity from Frequency Count (SubtlexUS)
            res.update(TTRF)
            res.update(VarF)
            res.update(PsyF)
            res.update(WoLF)

        # Shallow Traditional (ShTra) Features
        if 'LxSem' in self.args.linguistic_features:
            ShaF = LingFeat.ShaF_()  # Shallow Features (e.g. avg number of tokens)
            TraF = LingFeat.TraF_()  # Traditional Formulas
            res.update(ShaF)
            res.update(TraF)
        return res

    def extract_features(self, texts):
        """
        :param texts:
        :return:
                - df: Dataframe
                - counts: Dict (key: str, value: List of each answer's feature)
        """
        if self.args.feature_type == 'knowledge':
            # UIE, for opinion sentiment and NER
            aggregated_features = self.extract_knowledge_features(texts)
        elif self.args.feature_type == 'linguistic':
            # linguistic
            aggregated_features = self.extract_linguistic_features(texts)
        elif self.args.feature_type == 'classify':
            # classify by HC3
            aggregated_features = self.extract_classify_features(texts)
        else:
            # aggregate
            knowledge_features = self.extract_knowledge_features(texts)
            linguistic_features = self.extract_linguistic_features(texts)
            classify_features = self.extract_classify_features(texts)
            aggregated_features = {**knowledge_features, **linguistic_features, **classify_features}
        # print(aggregated_features)

        return aggregated_features

    def get_HC3_classify_bot_feature(self, q, a, lang='en'):
        # score by classifier
        q_sp = q.split()
        if len(q_sp) > 125:
            q = " ".join(q_sp[:125])
        a_sp = a.split()
        if len(a_sp) > 325:
            a = " ".join(a_sp[:325])
        payload = {
            "q": q,
            "a": a,
            "language": lang,
        }
        # response = requests.post(CONFIG.classification_api, json=payload)
        # classifier_score = response.json()
        try:
            classifier_score = self.detector.get_classification(payload)
        except Exception:
            classifier_score = {
                'prob': [100] * 4,
                'label': ['ChatGPT'] * 4,
                'method': ['qa', 'single', 'gltr', 'ppl']
            }
            # {'prob': prob, 'label': label}
            # q = " ".join(q_sp[:256])
            # payload = {
            #     "q": q,
            #     "a": a,
            #     "language": lang,
            # }
            # response = requests.post(CONFIG.classification_api, json=payload)
            # classifier_score = response.json()
            print("payload for wrong")
            print(payload)

        probs = classifier_score['prob']
        labels = classifier_score['label']
        methods = classifier_score['method']
        bot_features = []
        for i, method in enumerate(methods):
            prob = probs[i]
            label = labels[i]
            if label == "human":
                bot_feature = 100 - prob
            else:
                bot_feature = prob
            bot_features.append(bot_feature)
        return bot_features, methods
