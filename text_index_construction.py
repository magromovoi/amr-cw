import os
import re
import shutil
import pandas as pd
from utils import human_sort


def construct_text_indices(dataset, classes, raw_texts_prefix, raw_texts_csv_prefix):

    d = {'custom_index': [], 'text': [], 'label': [], 'split': [], 'target': []}

    encode_dict = {}

    for class_index, class_name in enumerate(classes):
        encode_dict[class_name] = class_index

    split_pattern = "(train|test)"

    for class_name in classes:
        index_pattern = fr"{class_name}_post_(train|test)_([0-9])*"
        class_pattern = fr"{class_name}"

        class_texts_path = raw_texts_prefix + '/' + class_name

        text_file_names = [os.path.join(class_texts_path, text_file_name)
                           for text_file_name in os.listdir(class_texts_path)]

        text_file_names = human_sort(text_file_names)

        class_index_column = [re.search(index_pattern, text_file_name).group() for text_file_name in text_file_names]
        class_index_column = [custom_index.replace('_post_', '_') for custom_index in class_index_column]
        class_text_column = [open(text_file_name, 'r').read() for text_file_name in text_file_names]
        label_column = [re.search(class_pattern, text_file_name).group() for text_file_name in text_file_names]
        split_column = [re.search(split_pattern, text_file_name).group() for text_file_name in text_file_names]
        target_column = [encode_dict[label_column_ele] for label_column_ele in label_column]

        d['custom_index'].extend(class_index_column)
        d['text'].extend(class_text_column)
        d['label'].extend(label_column)
        d['split'].extend(split_column)
        d['target'].extend(target_column)

    texts_df = pd.DataFrame(data=d)
    texts_df_file_name = f'{dataset}_raw_texts.csv'
    texts_df_file_path = raw_texts_csv_prefix + '/' + texts_df_file_name

    texts_df.to_csv(texts_df_file_name, index=False)
    shutil.move(texts_df_file_name, texts_df_file_path)
