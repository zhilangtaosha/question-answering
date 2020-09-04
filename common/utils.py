import json


def save_json(obj_to_save, output_filename):
    with open(output_filename, 'w', encoding='utf8') as out:
        json.dump(obj_to_save, out, ensure_ascii=False)
