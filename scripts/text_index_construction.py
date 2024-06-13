import os
import re
import shutil
import pandas as pd
from itertools import chain


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


def construct_text_indices(dataset, classes, text_prefix, csv_prefix):

    d = {'index': [], 'text': [], 'class': [], 'split': [], 'target': []}

    encode_dict = {}

    for class_index, class_name in enumerate(classes):
        encode_dict[class_name] = class_index

    split_pattern = "(train|test)"

    for class_name in classes:
        index_pattern = fr"{class_name}_post_(train|test)_([0-9])*"
        class_pattern = fr"{class_name}"

        class_texts_path = text_prefix + '/' + class_name

        text_file_names = [os.path.join(class_texts_path, text_file_name)
                           for text_file_name in os.listdir(class_texts_path)]

        text_file_names.sort()

        class_index_column = [re.search(index_pattern, text_file_name).group() for text_file_name in text_file_names]
        class_index_column = [index.replace('_post_', '_') for index in class_index_column]
        class_text_column = [open(text_file_name, 'r').read() for text_file_name in text_file_names]
        class_column = [re.search(class_pattern, text_file_name).group() for text_file_name in text_file_names]
        split_column = [re.search(split_pattern, text_file_name).group() for text_file_name in text_file_names]
        target_column = [encode_dict[class_column_ele] for class_column_ele in class_column]

        d['index'].append(class_index_column)
        d['text'].append(class_text_column)
        d['class'].append(class_column)
        d['split'].append(split_column)
        d['target'].append(target_column)

    d['index'] = flatten_chain(d['index'])
    d['text'] = flatten_chain(d['text'])
    d['class'] = flatten_chain(d['class'])
    d['split'] = flatten_chain(d['split'])
    d['target'] = flatten_chain(d['target'])

    df = pd.DataFrame(data=d)
    dataframe_file_name = f'{dataset}.csv'
    dataframe_file_path = csv_prefix + '/' + dataframe_file_name

    df.to_csv(dataframe_file_name, index=False)
    shutil.move(dataframe_file_name, dataframe_file_path)

    print("Break point.")
