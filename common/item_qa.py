import json


class ItemQA:
    """
    This class is mainly used in the data preprocessing scripts,
    for producing preprocessed datasets.
    """
    def __init__(self, question_id=None, question=None, **kwargs):
        self.question_id = question_id
        self.question = question
        self.answers = set()
        for attr in kwargs.keys():
            self.__dict__[attr] = kwargs[attr]

    def set_question_id(self, question_id):
        self.question_id = question_id

    def set_question(self, question):
        self.question = question

    def add_answer(self, answer):
        if answer:
            self.answers.add(answer)

    def json(self):
        result = self.__dict__.copy()
        result['answers'] = list(self.answers)
        return result

    def json_string(self):
        return json.dumps(self.json())


class ItemQA2:
    """
    This class is mainly used in the evaluation run scripts,
    for producing evaluation outputs.
    """
    def __init__(self, question_id=None, question=None, **kwargs):
        self.question_id = question_id
        self.question = question
        self.pred_answers = set()
        self.gold_answers = set()
        for attr in kwargs.keys():
            self.__dict__[attr] = kwargs[attr]

    def set_question_id(self, question_id):
        self.question_id = question_id

    def set_question(self, question):
        self.question = question

    def add_pred_answer(self, answer):
        if answer:
            self.pred_answers.add(answer)

    def add_gold_answer(self, answer):
        if answer:
            self.gold_answers.add(answer)

    def json(self):
        result = self.__dict__.copy()
        result['pred_answers'] = list(self.pred_answers)
        result['gold_answers'] = list(self.gold_answers)
        return result

    def json_string(self):
        return json.dumps(self.json())
