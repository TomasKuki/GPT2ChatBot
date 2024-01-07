import os
from nltk import sent_tokenize
from bs4 import BeautifulSoup

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
    return text

def process_ebook(file_path, start_page=50, end_page=50):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')

    # Skip the first 'start_page' pages and the last 'end_page' pages
    paragraphs = paragraphs[start_page:-end_page]

    text_content = ' '.join([paragraph.get_text() for paragraph in paragraphs])
    sentences = sent_tokenize(text_content)

    formatted_lines = ['{};'.format(sentence.replace('\n', ' ')) for sentence in sentences]
    return formatted_lines

def main(folder_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as output:
        for filename in os.listdir(folder_path):
            if filename.endswith(".html"):
                file_path = os.path.join(folder_path, filename)
                lines = process_ebook(file_path)
                output.write('\n'.join(lines))
                output.write('\n')

if __name__ == "__main__":
    folder_path = "./ebooks"
    output_file = "chatDataSet.txt"
    main(folder_path, output_file)
