import sys
import rapidjson as json

from tqdm import tqdm


from rubyslippers import extract_pages_from_dump
from rubyslippers import WikiExtractor

we = WikiExtractor('en')

output_dir = sys.argv[1]

for idx, page in tqdm(enumerate(extract_pages_from_dump(sys.stdin))):

    wiki_page = we.extract(page)
    if wiki_page:
        if wiki_page['text']:
            with open(output_dir + 'dump.txt', "a") as f:
                for s in wiki_page['text']:
                    f.write(s + '\n')
                f.write('\n')
