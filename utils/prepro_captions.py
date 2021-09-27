"""Modified from https://github.com/lukemelas/image-paragraph-captioning
"""
import os
import json
from collections import OrderedDict

# Caption data directory
data = './data/stanfordParagraph/paragraphs/'

# Files
para_json = os.path.join(data, 'paragraphs_v1.json')
train_split = os.path.join(data, 'train_split.json')
val_split = os.path.join(data, 'val_split.json')
test_split = os.path.join(data, 'test_split.json')

# Output files (should be coco-caption directory)
coco_captions = 'data/coco_gt'
train_outfile = os.path.join(coco_captions, 'para_captions_train.json')
val_outfile = os.path.join(coco_captions, 'para_captions_val.json')
test_outfile = os.path.join(coco_captions, 'para_captions_test.json')
assert os.path.exists(coco_captions)

# Load data
para_json = json.load(open(para_json))
train_split = json.load(open(train_split))
val_split = json.load(open(val_split))
test_split = json.load(open(test_split))

# Next, we tokenize the paragraphs and reformat them to a JSON file
# with the format expected by coco-captions
train = {
    'images': [],
    'annotations': [],
    'info': {'description': 'Visual genome paragraph dataset (train split)'},
    'type': 'captions',
    'licenses': 'http://creativecommons.org/licenses/by/4.0/',
}

val = {
    'images': [],
    'annotations': [],
    'info': {'description': 'Visual genome paragraph dataset (val split)'},
    'type': 'captions',
    'licenses': 'http://creativecommons.org/licenses/by/4.0/',
}

test = {
    'images': [],
    'annotations': [],
    'info': {'description': 'Visual genome paragraph dataset (test split)'},
    'type': 'captions',
    'licenses': 'http://creativecommons.org/licenses/by/4.0/',
}

# replacement: same with preprocess.py
replacements = OrderedDict({
    u'½': u'half',
    u'—': u'-',
    u'™': u'',
    u'¢': u'cent',
    u'ç': u'c',
    u'û': u'u',
    u'é': u'e',
    u'°': u' degree',
    u'è': u'e',
    u'…': u'',
    '.T': '. T',
    '.th': '. th',
    '.A': '. A',
    '.a': '. a',
    't.v.': 'tv',
    '...': '. ',
    '..': '. ',
})

# Loop over images
unique_ids = []
for imgid, item in enumerate(para_json):

    # Log progress
    if imgid % 1000 == 0:
        print('{}/{}'.format(imgid, len(para_json)))

    # Extract info
    url           = item['url']                    # original url
    filename      = item['url'].split('/')[-1]     # filename also is: str(item['image_id']) + '.jpg'
    id            = item['image_id']               # visual genome image id (filename)
    raw_paragraph = item['paragraph']
    split         = train if id in train_split else (val if id in val_split else test)

    # Skip duplicate paragraph captions
    if id in unique_ids:
        continue
    else:
        unique_ids.append(id)

    # Extract image info
    image = {
        'url': item['url'],
        'file_name': filename,
        'id': id,
    }

    for k, v in replacements.items():
        raw_paragraph = raw_paragraph.replace(k, v)

    # Extract caption info
    annotation = {
        'image_id': id,
        'id': imgid,
        'caption': raw_paragraph
    }

    # Store info
    split['images'].append(image)
    split['annotations'].append(annotation)

print('Finished converting to coco-captions format.')
print('There are {} duplicate captions.'.format(len(para_json) - len(unique_ids)))
print('The {} split contains {} images and {} annotations'.format('train', len(train['images']), len(train['annotations'])))
print('The {} split contains {} images and {} annotations'.format('val', len(val['images']), len(val['annotations'])))
print('The {} split contains {} images and {} annotations'.format('test', len(test['images']), len(test['annotations'])))

# Save files
for split, fname in [(train, train_outfile), (val, val_outfile), (test, test_outfile)]:
    with open(fname, 'w') as f:
        json.dump(split, f)
