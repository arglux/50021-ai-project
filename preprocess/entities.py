import pandas as pd
from headers import *
def extract_entites(data):
    entities = data['Entities'].str.split(";")
    entity_no = []
    for ent in entities:
        ent.pop()
        if ent[0] == 'null':
            entity_no.append(0)
        else:
            entity_no.append(len(ent))
    data['No. of Entities'] = entity_no
    return data

if __name__ == '__main__':

    data = pd.read_csv("../data/TweetsCOV19_052020.tsv.gz", compression='gzip', names=headers, sep='\t', quotechar='"')
    new_data = extract_entites(data)