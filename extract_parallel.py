
import sys

from joblib import Parallel, delayed
from tqdm import tqdm

from rubyslippers import extract_pages_from_dump
from rubyslippers import WikiExtractor

def parallelize_preprocess(func, iterator, processes, progress_bar=False):
    iterator = tqdm(iterator) if progress_bar else iterator
    if processes <= 1:
        return map(func, iterator)
    return Parallel(n_jobs=processes)(delayed(func)(line) for line in iterator)

processes = 20
quiet = True
we = WikiExtractor('fr')

output_dir = sys.argv[1]

pages = []
for idx, page in tqdm(enumerate(extract_pages_from_dump(sys.stdin))):
    pages.append(page)
    if len(pages) % processes == 0:
        for wiki_page in parallelize_preprocess(
            we.extract, pages, processes, progress_bar=(not quiet)
        ):
            if wiki_page:
                if wiki_page['text']:
                    with open(output_dir + 'dump.txt', "a") as f:
                        for s in wiki_page['text']:
                            f.write(s + '\n')
                        f.write('\n')

        pages = []

if len(pages) > 0:
    for wiki_page in parallelize_preprocess(
        we.extract, pages, processes, progress_bar=(not quiet)
    ):
        if wiki_page:
            if wiki_page['text']:
                with open(output_dir + 'dump.txt', "a") as f:
                    for s in wiki_page['text']:
                        f.write(s + '\n')
                    f.write('\n')
