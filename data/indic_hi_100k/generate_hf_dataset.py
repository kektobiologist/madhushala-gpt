from datasets import Dataset
import os
def gen():
    with open(os.path.join(os.path.dirname(__file__), 'hi_100k.txt'), "r") as g:
        # read line by line
        for line in g:
            if line.strip():
                yield {'text': line}
ds = Dataset.from_generator(gen)
# save to file
ds.save_to_disk(os.path.join(os.path.dirname(__file__), 'hi_100k_hf'))