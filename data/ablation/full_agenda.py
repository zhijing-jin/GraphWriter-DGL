import json
from itertools import permutations
from efficiency.log import fwrite

REL = ' -- RELATION -- '
src_f_templ = 'data/agenda/{}.json'
tgt_f_templ = 'data/agenda/full_{}.json'
splits = 'train valid test'.split()
for split in splits:
    src_f = src_f_templ.format(split)
    tgt_f = tgt_f_templ.format(split)
    with open(src_f) as f:
        data = json.load(f)
    for line in data:
        entities = line['entities']
        combinations = list(permutations(entities, 2))
        line['relations']=[e0 + REL + e1 for e0, e1 in combinations]

    print('[Info] Saving {} data from {} to {}'.format(len(data), src_f, tgt_f))

    fwrite(json.dumps(data, indent=4), tgt_f)

