# import re
import regex as re
import urllib.parse
from itertools import chain

from rubyslippers.utils import get_wiki_document_url
from rubyslippers.utils import unescape_xml

from nltk.tokenize import sent_tokenize


class WikiExtractor:
    GARBAGE_TAGS = ('ref', 'gallery', 'timeline', 'noinclude', 'pre', 'table',
                    'tr', 'td', 'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'menu', 'dir')

    WRAPPER_TAGS = ('nowiki', 'cite', 'source', 'hiero', 'div', 'font', 'span',
                    'strong', 'strike', 'blockquote', 'tt', 'var', 'sup', 'sub', 'big',
                    'small', 'center', 'h1', 'h2', 'h3', 'em', 'b', 'i', 'u', 'a', 's', 'p')

    SINGLE_TAGS = ('references', 'ref', 'img', 'br', 'hr', 'li', 'dt', 'dd')

    PLACEHOLDER_TAGS = {'math': 'formula', 'code': 'codice'}

    PROJECT_NAMESPACES = ('wikipedia', 'mediawiki', 'wikiquote', 'wikibooks',
                          'wikisource', 'wiktionary', 'wikispecies', 'wikinews', 'wikiversita',
                          'commons')

    GARBAGE_NAMESPACES = ('immagine', 'image',
                          'categoria', 'category',  # Maybe this needs to be disabled.
                          'file')

    # Recognize HTML comments.
    COMMENT_PATTERN = [re.compile(r'<!--.*?-->', re.DOTALL)]

    # Recognize garbage HTML tags.
    GARBAGE_TAG_PATTERNS = [
        re.compile(fr'<\s*{tag}(\s*| [^/]+?)>.*?<\s*/\s*{tag}\s*>',
                   re.DOTALL | re.IGNORECASE)
        for tag in GARBAGE_TAGS
    ]

    # Recognize HTML wrapper tags.
    WRAPPER_TAG_PATTERNS = []
    for tag in WRAPPER_TAGS:
        left_pattern = re.compile(fr'<\s*{tag}(\s*| [^/]+?)>', re.DOTALL | re.IGNORECASE)
        right_pattern = re.compile(fr'<\s*/\s*{tag}\s*>', re.DOTALL | re.IGNORECASE)
        WRAPPER_TAG_PATTERNS.append(left_pattern)
        WRAPPER_TAG_PATTERNS.append(right_pattern)

    # Recognize single HTML tags.
    SINGLE_TAG_PATTERNS = []
    for tag in SINGLE_TAGS:
        good_pattern = re.compile(fr'<\s*{tag}(\s*| .+?)/\s*>', re.DOTALL | re.IGNORECASE)
        bad_pattern = re.compile(fr'<\s*(/|\\)?\s*{tag}(\s*| [^/]+?)\\?\s*>', re.DOTALL | re.IGNORECASE)
        SINGLE_TAG_PATTERNS.append(good_pattern)
        SINGLE_TAG_PATTERNS.append(bad_pattern)

    # Recognize HTM placeholder tags.
    PLACEHOLDER_TAG_PATTERNS = []
    for tag, replacement in PLACEHOLDER_TAGS.items():
        pattern = re.compile(fr'<\s*{tag}(\s*| [^/]+?)>.*?<\s*/\s*{tag}\s*>', re.DOTALL | re.IGNORECASE)
        PLACEHOLDER_TAG_PATTERNS.append((pattern, replacement))

    # Recognizes tables and templates.
    RECURSIVE_TABLE_PATTERN = re.compile(r'(?<table>\{(?:[^{}]+|(?&table))*\})', re.DOTALL | re.IGNORECASE)
    TABLE_PATTERN = re.compile(r'([^|]*?)\|([^}]*)|(formatnum):([^}]*)', re.DOTALL | re.IGNORECASE)

    # Recognize wikilinks.
    GOOD_WIKILINK_PATTERN = re.compile(r'(?<link>\[(?:[^\[\]]{1,1000}+|(?&link))*\])', re.DOTALL)
    BAD_LEFT_WIKILINK_PATTERN = re.compile(r'\[[^[]*?\]\]', re.DOTALL)
    BAD_RIGHT_WIKILINK_PATTERN = re.compile(r'\[\[[^[]*?\]', re.DOTALL)

    INFO_TYPE = ['s-', 'lang', 'date de naissance', 'unité',
                 'coord', 'p.', 'date-', 'nombre', 'nobr', 'msapi',
                 'citation', 'api', 'date de décès', 'durée', 'date',
                 'date de mort', '-millénaire', 'math']

    LINK_PATTERN = re.compile(r'<a href=\"([^>]*)\"[^>]*>(.*?)<\/a>', re.DOTALL | re.IGNORECASE)  # <a href=\"([^>]*)\"[^>]*>([^>]*(?R)?)<\/a>
    BAD_LINK_PATTERN = re.compile(r'<a href=\"([^>]*)\"[^>]*>', re.DOTALL | re.IGNORECASE)
    # RESIDUAL_HTML_BLOCKS = re.compile(r'\<(?!\/)(gallery|comment|imagemap) ?.{0,100}?\>(.|\n){1,1000}?\<\/(\1)\>', re.DOTALL | re.IGNORECASE)  # imagemap, gallery mode, comments ...

    HTTP_LINK_PATTERN = re.compile(r'\[http.*?\]', re.DOTALL | re.IGNORECASE)
    RESIDUAL_FILES = re.compile(r'Fichier:.*', re.DOTALL | re.IGNORECASE)

    # Recognizes apostrophes that precede bold and italics.
    APOSTROPHE_BOLD_PATTERN = re.compile(r"\w'('''[^\s'][^']*?[^\s']''')[^']", re.DOTALL)
    APOSTROPHE_ITALICS_PATTERN = re.compile(r"\w'(''[^\s'][^']*?[^\s']'')[^']", re.DOTALL)

    # Recognizes numeric entities.
    NUMERIC_ENTITY_PATTERN = re.compile(r'&#\d+?;')

    # Recognizes multiple spaces.
    MULTI_SPACE_PATTERN = re.compile(r' {2,}')

    # Recognizes multiple dots.
    MULTI_DOT_PATTERN = re.compile(r'\.{4,}')

    # List of patterns that are in the cleaning function.
    PATTERNS_FOR_DELETION = [
        COMMENT_PATTERN,
        GARBAGE_TAG_PATTERNS,
        WRAPPER_TAG_PATTERNS,
        SINGLE_TAG_PATTERNS
    ]

    PATTERN_TOKENS = re.compile(
        r"(\->|(?::\)|:-\)|:\(|:-\(|;\);-\)|:-O|8-|:P|:D|:\||:S|:\$|:@|8o\||\+o\(|\(H\)|\(C\)|\(\?\))|(?:[\d.,]+)|([^\s\w0-9])\2*|(?:[\w0-9\.\-]+['’]?)(?<!\.))")

    def __init__(self, lang='fr'):
        self.prefix = f'http://{lang}.wikipedia.org/wiki/'

    def _handle_wikilink(self, wikilink):
        """
        From https://github.com/jodaiber/Annotated-WikiExtractor/blob/master/annotated_wikiextractor/wikiextractor.py#L378
        """
        tokens = wikilink.split(':')
        while not tokens[0]:
            if len(tokens) < 2: return '', ''
            tokens = tokens[1:]

        if len(tokens) == 1 or tokens[0].strip().lower() in self.PROJECT_NAMESPACES:
            tokens = tokens[-1].split('|')
            while not tokens[-1]:
                if len(tokens) < 2: return '', ''
                tokens = tokens[:-1]
            link_text = tokens[-1].split('#')[-1].split('/')[-1].strip()
            if len(tokens) > 1:
                article_title = tokens[-2].strip()
            else:
                article_title = link_text
            return article_title, link_text

        if tokens[0].strip().lower() in self.GARBAGE_NAMESPACES:
            return '', ''

        tokens = tokens[-1].split('|')
        while not tokens[-1]:
            if len(tokens) < 2: return '', ''
            tokens = tokens[:-1]
        if len(tokens) == 1: return '', ''
        link_text = tokens[-1].split('#')[-1].split('/')[-1].strip()
        if len(tokens) > 1:
            article_title = tokens[-2].strip()
        else:
            article_title = link_text
        return article_title, link_text

    def _get_anchor_tag(self, document_title, link_text):
        if not link_text:
            return ''
        if not document_title:
            return link_text
        wiki_url = get_wiki_document_url(document_title, '')
        return fr'<a href="{wiki_url}">{link_text}</a>'

    def _handle_unicode(self, entity):
        numeric_code = int(entity[2:-1])
        if numeric_code >= 0x10000:
            return ''
        return chr(numeric_code)

    def replace_tables(self, match):

        if match.group(1):
            match_type = match.group(1)
            value = match.group(2)
        else:
            match_type = match.group(3)
            value = match.group(4)

        if match_type is None:
            return ''
        elif match_type.lower() in self.INFO_TYPE:
            return re.sub('\|', ' ', value)
        elif len(match_type.lower().split()) and (match_type.lower().split()[-1] in self.INFO_TYPE):
            return re.sub('\|', ' ', value)
        elif match_type.lower() in 'date':
            return re.sub('.*?\|texte=(.*)', r'\1', value)
        else:
            return ''

    def recursive_replace_tables(self, match, depth=0):
        match = match.group('table').rstrip('}').lstrip('{')
        if self.RECURSIVE_TABLE_PATTERN.search(match) and depth < 5:
            match = self.RECURSIVE_TABLE_PATTERN.sub(lambda x: self.recursive_replace_tables(x, depth + 1),
                                                     match)
        match = self.TABLE_PATTERN.sub(self.replace_tables, match)
        return match

    def replace_wikilinks(self, match):
        wikilink = match.group()
        if re.match(r'\[{1,2}Fichier\:', wikilink):
            return ''
        document_title, link_text = self._handle_wikilink(wikilink[2:-2])
        anchor_tag = self._get_anchor_tag(document_title, link_text)
        return anchor_tag

    def recursive_replace_wikilinks(self, text, depth=0):
        """Handle good wikilinks (well formatted; two nesting levels)"""
        text = re.sub(self.GOOD_WIKILINK_PATTERN, self.replace_wikilinks, text)
        if self.GOOD_WIKILINK_PATTERN.search(text) and depth < 10:
            text = self.recursive_replace_wikilinks(text, depth + 1)
        return text

    def clean(self, text):
        # Unescape greater/lesser than symbols.
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('<<', u'��').replace('>>', u'��')

        # Delete tags.
        for pattern in chain(*self.PATTERNS_FOR_DELETION):
            text = pattern.sub('', text)

        # Substitute placeholder tags.
        for idx, (pattern, placeholder) in enumerate(self.PLACEHOLDER_TAG_PATTERNS, start=1):
            for match in pattern.finditer(text):
                text = text.replace(match.group(), f'{placeholder}_{idx}')

        # Deletes tables and tempaltes.
        text = text.replace('{{end box}}', '}')
        text = text.replace('{{', '{').replace('}}', '}')
        text = text.replace('{|', '{').replace('|}', '}')
        text = self.RECURSIVE_TABLE_PATTERN.sub(self.recursive_replace_tables, text)
        text = self.recursive_replace_wikilinks(text)

        # Handles bad wikilinks (poorly formatted)
        for match in self.BAD_LEFT_WIKILINK_PATTERN.finditer(text):
            wikilink = match.group()
            document_title, link_text = self._handle_wikilink(wikilink[1:-2])
            text = text.replace(wikilink, self._get_anchor_tag(document_title, link_text))

        for match in self.BAD_RIGHT_WIKILINK_PATTERN.finditer(text):
            wikilink = match.group()
            document_title, link_text = self._handle_wikilink(wikilink[2:-1])
            text = text.replace(wikilink, self._get_anchor_tag(document_title, link_text))

        text = text.replace('[[', '').replace(']]', '')

        # Handles bold and italics.
        for match in self.APOSTROPHE_BOLD_PATTERN.finditer(text):
            bold_text = match.group(1)
            text = text.replace(bold_text, bold_text[3:-3])
        for match in self.APOSTROPHE_ITALICS_PATTERN.finditer(text):
            italic_text = match.group(1)
            italic_text_replace = italic_text[2:-2]
            text = text.replace(italic_text, f'&quot;{italic_text_replace}&quot;')
        text = text.replace("'''", '').replace("''", '&quot;')

        # Handles XML special characters.
        text = text.replace('&amp;', '&').replace('&quot;&quot;', '&quot;')
        text = unescape_xml(text)

        # Handles special numeric entities.
        for match in self.NUMERIC_ENTITY_PATTERN.finditer(text):
            entity = match.group()
            text = text.replace(entity, self._handle_unicode(entity))

        # Handles some imperfections in the text.
        text = text.replace('\t', ' ')
        text = self.MULTI_SPACE_PATTERN.sub(' ', text)
        text = self.MULTI_DOT_PATTERN.sub('...', text)
        text = text.replace(' ,', ',').replace(' .', '.')
        text = text.replace(' :', ':').replace(' ;', ';')
        text = text.replace(',,', ',').replace(',.', '.')
        text = text.replace('( ', '(').replace(' )', ')')
        text = text.replace('[ ', '[').replace(' ]', ']')
        text = text.replace(u'�� ', u'��').replace(u' ��', u'��')

        return text

    def compact(self, text, min_count=6):
        """ Adapted from https://github.com/jodaiber/Annotated-WikiExtractor/blob/master/annotated_wikiextractor/wikiextractor.py#L330 """
        page = list()
        paragraph = list()

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Handles the title of the page
            if line.startswith('++'):
                title = line[2:-2]
                if title and title[-1] not in '!?':
                    title = fr'{title}.'
                page = [title]
            # Handles paragraph titles.
            elif re.match(r'={2,}', line): # line.startswith('=='):
                if len(paragraph) > 1:
                    page.extend(paragraph)
                if re.findall(r'={2,} *([^=]*)', line):
                    title = re.findall(r'={2,} *([^=]*)', line)[0].strip()
                else:
                    title = line[2:-2]
                if title and title[-1] not in '!?':
                    title = fr'{title}.'
                paragraph = [title]

            # Remove bulleted and numbered lists.
            # TODO: Do we really want to remove them?
            elif line[-1] == ':' or line[0] in '*#:;':
                continue

            # Remove the remnants of the tables.
            elif line[0] in '{|' or line[-1] in '}':
                continue

            # Remove irrelevant lines.
            elif (line[0] == '(' and line[-1] == ')') or line.strip('.-') == '':
                continue

            # Eliminate lines with a low token count.
            elif not '_' in line and len(line.split()) < min_count:
                continue
            # Handles the page.
            elif len(paragraph) == 0:
                page.append(line)
            # handles the paragraph.
            else:
                paragraph.append(line)

        # Skip title only paragraph.
        if len(paragraph) >= 1:
            page.extend(paragraph)

        # Skip title only page.
        elif len(page) == 0:
            return ''

        return '\n'.join(page)

    def sentence_tokenize(self, text):
        sentences = sent_tokenize(text)
        # sentences = re.split(self.PATTERN_SENTENCES, text)
        return sentences

    def word_tokenize(self, text):
        if not isinstance(text, list):
            text = [text]
        tokens = [[groups[0] for groups in re.findall(self.PATTERN_TOKENS, s)] for s in text]
        tokens = [' '.join([t.strip() for t in s]) for s in tokens]
        return tokens

    def parse(self, page):
        wiki_id = None
        wiki_url = None
        wiki_text = []
        wiki_infoboxes = []
        this_infobox = []
        residual_html = []

        # `page` is the XML <page> ... </page>
        for line in page:
            if not line.strip():
                continue
            # Page identifier <id> ... </id>
            if not wiki_id and line.startswith('<id>') and line.endswith('</id>'):
                wiki_id = int(line[4:-5])
                continue
            # Page title <title> ... </title>
            elif not wiki_url and line.startswith('<title>') and line.endswith('</title>'):
                title = line[7:-8].replace('&amp;', '&')
                if ':' in title:
                    break
                wiki_url = get_wiki_document_url(title, self.prefix)
                wiki_url = get_wiki_document_url(title, self.prefix)
                text = '++{title}++'
                continue
            # Beginning of the page text.
            elif line.strip().startswith('<text'):
                if line.endswith('</text>'):
                    break
                # line = line[27:]
                line = re.sub(r"^<text.*?>", '', line)
                if not line.strip():
                    continue
            # End of the page text.
            elif line.endswith('</text>'):
                line = line[:-7]
                if not line.strip():
                    continue

            # Paragraph title.
            elif line[0] == '=':
                continue

            # Keeps track of infoboxes.
            if this_infobox == [] and line.startswith('{{Infobox '):
                this_infobox.append(line)
            elif this_infobox and line.startswith('|'):
                this_infobox.append(line)
            elif this_infobox and line == '}}':
                this_infobox.append(line)
                # Add infobox to this page.
                wiki_infoboxes.append('\n'.join(this_infobox))
                # Pop out the storage for `this_infobox`
                this_infobox = []

            # Keeps track of residual html
            if re.match(r'(?:\<|&lt;)(?!\/)(gallery|comment|imagemap) ?.{0,100}?(?:\>|&gt;)', line.strip()):
                residual_html.append(line)
            if residual_html:
                if re.search(r'(?:\<|&lt;)\/(gallery|comment|imagemap) ?.{0,100}?(?:\>|&gt;)', line.strip()):
                    residual_html = []
                continue
            # Superfluous information.
            if line[0] == '<':
                continue

            wiki_text.append(line)

        if wiki_text:
            return wiki_id, wiki_url, wiki_text, wiki_infoboxes

    def extract(self, page):
        # start = time.time()
        # Call the parsing function.
        # remove residual html

        parsed_page = self.parse(page)
        if parsed_page:
            wiki_id, wiki_url, wiki_text, wiki_infoboxes = parsed_page

        # Only returns when wiki_text is non-empty.
        if parsed_page and wiki_text:
            # Preparing to process and return the wiki page json.
            wiki_cats, annotations, infobox_types = [], [], []

            # Extract infobox types.
            infobox_types = [box.split('\n')[0].partition(' ')[2]
                             for box in wiki_infoboxes]

            # Convert page texts to single string.
            wiki_text = self.compact(self.clean('\n'.join(wiki_text)))

            # Extract the categories.
            for match in re.finditer(r"\[\[Category:(.*)\]\]", wiki_text):
                for cat in match.groups(0):
                    wiki_cats.append(cat.strip().strip('|'))

            # Extract the annotations.
            # Remove the <a href="..."> ... </a>
            wiki_text, annotations = self.annotate(wiki_text)

            wiki_text = self.sentence_tokenize(wiki_text)
            wiki_text = [s for s in wiki_text if not self.RESIDUAL_FILES.search(s)]
            # wiki_text = self.word_tokenize(wiki_text)
            # Put everything into a serializable dictionary.
            wiki_page = {'id': wiki_id, 'url': wiki_url, 'text': wiki_text,
                         'categories': wiki_cats, 'infobox_types': infobox_types,
                         'annotations': annotations}
            return wiki_page

    def replace_links(self, text, depth=0):
        text = re.sub(self.LINK_PATTERN, r'\2', text)
        if re.search(self.LINK_PATTERN, text) and depth < 5:
            text = self.replace_links(text, depth=depth+1)
        return text

    def annotate(self, text, keep_anchors=False):
        # Initialize an empty annotations list.
        annotations = []

        # This int is used to keep track of the difference between the original
        # article with <a href=".."> links and the new article that only
        # contains the label of the link.
        deltaStringLength = 0

        # As a first step, find all links in the article, save their positions
        # into the annotations object
        ms = re.finditer('<a href=\"([^\"]+)\">([^>]+)<\/a>', text)

        for m in ms:
            if urllib.parse.quote("#") not in m.group(1) or keep_anchors:
                annotations.append({
                    "uri": m.group(1),
                    "surface_form": m.group(2),
                    "offset": m.start() - deltaStringLength
                })

            deltaStringLength += len(m.group(0)) - len(m.group(2))

        # As a second step, replace all links in the article by their label
        text = self.replace_links(text)
        text = re.sub(self.BAD_LINK_PATTERN, '', text)
        return text, annotations
