import re
import json
import pandas as pd
from graph_dataset import GraphDataset
from ten_newsgroups_dataset import NewsGroupDataset

'''
docs = [["The Aviator has been tipped by UK bookmakers as the favourite to win the best film award at this years Oscars. "
         "Ray star Jamie Foxx is clear favourite in the best actor category while Million Dollar Babys Hilary Swank is tipped to win the best actress prize. "
         "Bookmakers predict Cate Blanchett will be named best supporting actress."],
        ["Hollywood legend Dustin Hoffman has hit out at the quality of current films and theatre productions. "
         "Hoffman also said he stopped working a few years ago and moved into directing and writing. "
         "There is no first or second act."],
        ["Star Trek fans have taken out a full-page ad in the Los Angeles Times in an attempt to persuade TV executives not to scrap Star Trek: Enterprise. "
         "They are also asking the Sci-Fi Channel to pick it up from UPN and will stage a rally in Los Angeles on 25 February. "
         "It also included a cut-out coupon for fans to send to UPNs parent companies Paramount and Viacom plus the Sci-Fi Channel. "]]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

print("Break point.")

for doc in docs:
    encoded_input = tokenizer(doc, padding=True, truncation=True, max_length=512,
                   return_token_type_ids=True)

    output = model(**encoded_input)
    last_hidden_state = output['last_hidden_state']
    cls_embedding = last_hidden_state[:, 0, :].detach().numpy()

    print("Break point.")


text_embeddings_file_name = '/home/eric/Eric/PhD/Research Code/gnn/data/embeddings/ten_newsgroups_text_embeddings.json'

with open(text_embeddings_file_name) as f:
    d = json.load(f)

text_embeddings_df = pd.DataFrame.from_dict(d)

print("Break point.")

GraphDataset(root='/home/eric/Eric/PhD/Research Code/gnn/data/datasets/ten_newsgroups_graphs_dataset', labels=['business', 'entertainment', 'food', 'graphics', 'historical', 'medical', 'politics', 'space', 'sport', 'technologie'],
             dataset='ten_newsgroups')

class_names = ['business', 'entertainment', 'food', 'graphics', 'historical', 'medical', 'politics', 'space', 'sport', 'technologie']

# Make a regex that matches if any of our regexes match.
# combined = "(" + ")|(".join(regexes) + ")"

index_pattern = fr"({"|".join(class_names)})_(train|test)_graph_([0-9])*"

print("Break point.")

# index_pattern = fr"{class_name}_post_(train|test)_([0-9])*"


if re.match(combined, mystring):
    print "Some regex matched!"
'''

graph_embeddings_file_name = '/home/eric/Eric/PhD/Research Code/gnn/data/embeddings/ten_newsgroups_graph_embeddings.json'

with open(graph_embeddings_file_name) as f:
    d = json.load(f)

graph_embeddings_df = pd.DataFrame.from_dict(d)

text_embeddings_file_name = '/home/eric/Eric/PhD/Research Code/gnn/data/embeddings/ten_newsgroups_text_embeddings.json'

with open(text_embeddings_file_name) as f:
    d = json.load(f)

text_embeddings_df = pd.DataFrame.from_dict(d)

print("Break point.")
