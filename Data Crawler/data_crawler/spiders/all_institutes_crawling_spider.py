import scrapy
import json
import os
import re
import time
import logging
import tempfile
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from datetime import datetime
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from urllib.parse import urlparse, urljoin, unquote
from html import unescape
from unstructured.partition.pdf import partition_pdf
from langdetect import detect


################ HELPER FUNCTIONS ################
# Helper function to extract domain from URL
def get_domain(url):
    return url.split('/')[2]

# Function to sanitize directory names
def sanitize_dir_name(name):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name

def sanitize_file_name(name, max_length=255):
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    # Truncate the name if it exceeds the maximum length
    if len(name) > max_length:
        name = name[:max_length]
    return name

def clean_text(text):
    # Decode HTML entities in case any were missed
    text = unescape(text)
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    # Replace all special whitespaces with ' ' and remove extra
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (only allowing A-Z, a-z, 0-9, and some punctuation marks)
    text = re.sub(r'[^A-Za-z0-9.,!?;\'"()[\]{}\-:/_ ]+', '', text)
    # Remove multiple occurrences of punctuation
    text = re.sub(r'([.,!?;\'"()\-])\1+', r'\1', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    return text

# Extract the title from the text
def extract_title(text):
    # Remove the leading whitespaces
    text = text.lstrip()
    # Consider the first line as the title
    title = text.split('\n')[0].strip()
    return title

def extract_pdf_title(url):
    parsed_url = urlparse(url)
    file_name = parsed_url.path.split('/')[-1]
    return unquote(file_name[:-4])  # Remove .pdf extension


################ CRAWLER ################
class TUMInstitutesSpider(CrawlSpider):
    name = 'data_crawler'
    start_urls = ['https://www.tum.de']
    visited_urls = set()

    MAX_FILE_NAME_LENGTH = 100
    ROOT_DATA_FOLDER_PATH = os.getenv('CRAWLER_OUTPUT_DIR', 'crawling_results')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Output the configuration and wait for 5 seconds for the user to read it
    print(f"=== Log level set to: {LOG_LEVEL} ===")
    print(f"=== Crawling results will go to: {ROOT_DATA_FOLDER_PATH} ===")
    if not os.path.exists(ROOT_DATA_FOLDER_PATH):
        os.makedirs(ROOT_DATA_FOLDER_PATH)
    time.sleep(5)

    # Read the JSON data containing institutes and their links
    with open('institutes.json', 'r') as file:
        institutes_data = json.load(file)
        for institute in institutes_data:
            start_urls.append(institute['link'])

    # Define the domain to crawl
    allowed_domains = ['tum.de']

    # Rules to follow for crawling
    rules = (
        Rule(LinkExtractor(
            allow_domains=['tum.de']), callback='parse_html', follow=False),
            )

    def get_result_file_path(self, url):
        # Create folder to store the parsed data
        domain_folder = os.path.join(self.ROOT_DATA_FOLDER_PATH, sanitize_dir_name(get_domain(url)))
        os.makedirs(domain_folder, exist_ok=True)
        parsed_url = urlparse(url)
        url_parts = parsed_url.path.strip('/').split('/')
        url_parts = [part.split('.')[0] for part in url_parts]
        file_name = '_'.join(url_parts)
        if file_name == '':
            file_name = 'content'
        sanitized_name = sanitize_file_name(file_name, self.MAX_FILE_NAME_LENGTH)
        result_path = os.path.join(domain_folder, f"{sanitized_name}.json")
        return result_path

    def parse_pdf(self, response):
        # Download the PDF file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(response.body)
            tmp_file.close()
            tmp_filename = tmp_file.name
        
        # Extract text content from the PDF
        try:
            pdf_text = partition_pdf(tmp_filename)
        
            # Process the text content
            cleaned_text = " ".join([clean_text(element.text) for element in pdf_text])
            # Save data to JSON file named after the url parts
            page_path = self.get_result_file_path(response.url)
            page_data = {
                "url": response.url,
                "content": cleaned_text,
                "type": "pdf",
                "lastRetrievalTime": datetime.now().isoformat(),
                "title": extract_pdf_title(response.url) ,
                "language": detect(cleaned_text)
            }
            try:
                with open(page_path, 'w', encoding='utf-8') as page_file:
                    json.dump(page_data, page_file, indent=4)
            except Exception as e:
                logging.error(f"Error writing file {page_path}: {e}")
        except Exception as e:
            logging.error(f"Error processing PDF {response.url}: {e}")
        
        # Remove the temporary file after processing
        try:
            os.remove(tmp_filename)
        except OSError:
            pass
        

    # Function to parse items
    def parse_html(self, response):
        content_type = response.headers.get('Content-Type', b'').decode('utf-8')
        # Check if the response is HTML
        if 'text/html' not in content_type:
            return
        
        # Extracting text content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text()
        cleaned_text = clean_text(text_content)
        
        # Determine the language of the page
        parsed_url = urlparse(response.url)
        url_parts = parsed_url.path.strip('/').split('/')
        html_lang = 'en' if 'en' in url_parts else 'de'
        
        # Save data to JSON file named after the url parts
        page_path = self.get_result_file_path(response.url)
        page_data = {
            "url": response.url,
            "content": cleaned_text,
            "type": "html",
            "lastRetrievalTime": datetime.now().isoformat(),
            "title": extract_title(text_content) ,
            "language": html_lang
        }
        try:
            with open(page_path, 'w', encoding='utf-8') as page_file:
                json.dump(page_data, page_file, indent=4)
        except Exception as e:
            logging.error(f"Error writing file {page_path}: {e}")

        # Extract links from the page
        links = response.xpath('//a/@href').extract()
        # Extract and save files with different extensions
        for link in links:
            absolute_url = urljoin(response.url, link)
            if 'tum' in absolute_url and absolute_url.endswith('.pdf'):
                yield scrapy.Request(absolute_url, callback=self.parse_pdf)
