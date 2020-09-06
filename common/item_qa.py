import json


class ItemQA:
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
