import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF
import re

class WebScrapping:
    '''This class will handle web scrapping until we get html files. From a base url we extract all the available year links
    then extract the html files from the year links.'''
    def __init__(self, headers, base_url):
        # initialising required variables for scrapping website
        self.headers = headers
        self.base_url = base_url

    def get_year_links(self):
        '''
        This function extracts the years from the base url.
        :return: Returns a list of year links.
        '''
        response = requests.get(self.base_url, self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        year_links = []  # list that contains links of all the years available in base url.

        for tag in soup.find_all('a', href=True):
            href = tag['href']
            if re.match(r'/cases/new-jersey/tax-court/\d{4}/', href):
                # href contains part of the url eg: a -> <a href="/cases/new-jersey/tax-court/2025/">2025</a> ;
                # href: /cases/new-jersey/tax-court/2025/
                year_link = urljoin(self.base_url, href)
                # year_link contains https://law.justia.com/cases/new-jersey/tax-court/2025
                year_links.append(year_link)
        # year_links list has all the year links along with only 15 html links from 2025 and 2024 combined.
        # Removing html links from year_links to have only year links then access html links from all these links.
        filtered_year_links = []
        for link in year_links:
            if not link.endswith('html'):
                filtered_year_links.append(link)
        year_links = filtered_year_links

        #print(f"the length of year_links after removing html files are:{len(year_links)}")

        return year_links[:2] # for demo purpose

    def get_html_links(self, year_links):
        '''
        This program will access each year link from the list year_links and access all the html files from each year.
        :return: Returns a list of html files.
        '''

        html_links = []
        for link in year_links:
            response = requests.get(link, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            for tag in soup.find_all('a', href=True):
                href_links = tag['href']  # href_links contains hyperlinks from those links which contain href -> href - True
                if href_links.endswith('html'):
                    html_link = urljoin(link, href_links)
                    html_links.append(html_link)

        #print(f"the total number of html files read are: {len(html_links)}")  # 1299

        return html_links[:10] # for demo purpose










