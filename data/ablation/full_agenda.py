import json
from itertools import permutations
from efficiency.log import fwrite

rel_placeholder = ' -- RELATION -- '
splits = 'train valid test'.split()
file_templ = 'data/agenda/{}.json'
for split in splits:
    f_in = file_templ.format(split)
    f_emp = file_templ.format(split+'_empty')
    f_full = file_templ.format(split+'_full')
    with open(f_in) as f:
        data = json.load(f)

    data_empty = [{k: v if k != 'relations' else [] for k, v in item.items()}
                  for item in data]

    data_full = [{k: v for k, v in item.items() if k != 'relations'}
                 for item in data]
    for item, item_full in zip(data, data_full):
        entities = item['entities']
        entities = [e for e in entities if e]
        comb = list(permutations(entities, 2))
        item_full['relations'] = [rel_placeholder.join(c) for c in comb]

    data_empty[0]
    data_full[0]

    fwrite(json.dumps(data_empty, indent=4), f_emp)
    fwrite(json.dumps(data_full, indent=4), f_full)
