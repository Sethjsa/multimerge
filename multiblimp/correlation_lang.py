# use env: cuda11

import json
from distals import distals
model = distals.Distals()

with open('/home/saycock/multimerge/multiblimp/pairs.txt', 'r') as f:
    pairs = f.readlines()

language_map = {
    'eng': 'eng',
    'nld': 'nld',
    'spa': 'spa',
    'fra': 'fra',
    'rus': 'rus',
    'ita': 'ita',
    'tur': 'tur',
    'ara': 'ar',
    'deu': 'deu',
    'zhos': 'cmn',
}

distances = {}
for lang in language_map.values():
    distances[lang] = {}

# print(model.get_dists('zh', 'es', threshold=0.01))
# exit()

for lang in language_map.values():
    for lang2 in language_map.values():
        dists = model.get_dists(lang, lang2, threshold=0.01)
        # average_typology = np.mean(dists['typology']['phoible'], dists['typology']['grambank'], dists['typology']['glot_tree')
        # average_metadata = np.mean(dists['metadata']['nlp_state'], dists['metadata']['speakers'], dists['metadata']['AES'], dists['metadata']['loc'])
        # average_wordlists = np.mean(dists['wordlists']['asjp'], dists['wordlists']['concepts'])
        # average_textbased = dists['textbased']['textcat']
        distances[lang][lang2] = dists

with open('distances.json', 'w') as f:
    json.dump(distances, f)
