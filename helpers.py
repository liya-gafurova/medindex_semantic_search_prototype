from typing import Dict

from ebooklib import epub
from bs4 import BeautifulSoup, Tag


def parse_epub_content(epub_path: str) -> Dict:

    def paragraphize_node(node):
        return [' '.join(p.get_text().split()) for p in node.children if isinstance(p, Tag) and len(' '.join(p.get_text().split())) > 0]

    book = epub.read_epub(epub_path)

    article_name = book.title

    content = ''
    for item in book.items:
        if item.media_type and item.media_type == 'application/xhtml+xml':
            content += BeautifulSoup(item.content.decode("utf-8"), 'html5lib').get_text()

    sections = []
    main_content = BeautifulSoup(book.items[0].content.decode("utf-8"), 'html5lib')

    body = main_content.find('div', {'class': 'body'})
    for child in body.children:
        if isinstance(child, Tag):
            try:
                name = child.find('h2').get_text()
            except:
                name = ''

            section = dict(
                name=name,
                paragraphs=paragraphize_node(child)
            )

            sections.append(section)

    appendix_paragraphs = []
    for item in book.items[1:]:
        if item.media_type and item.media_type == 'application/xhtml+xml':
            appendix_body = BeautifulSoup(item.content.decode("utf-8"), 'html5lib').find('body')
            appendix_paragraphs.extend(paragraphize_node(appendix_body))

    appendix_paragraphs = [p for p in appendix_paragraphs if p != '[Back]']

    sections.append(dict(name='Appendix', paragraphs=appendix_paragraphs))

    return dict(
        article_name=article_name,
        sections=sections,
        content=content
    )
