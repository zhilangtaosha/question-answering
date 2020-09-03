import json


class ItemQA:
    def __init__(self, question_id=None, question=None):
        self.question_id = question_id
        self.question = question
        self.answers = set()

    def set_question_id(self, question_id):
        self.question_id = question_id

    def set_question(self, question):
        self.question = question

    def add_answer(self, answer):
        self.answers.add(answer)

    def json(self):
        return {
            'question_id': self.question_id,
            'question': self.question,
            'answers': list(self.answers)
        }

    def json_string(self):
        return json.dumps(self.json())
