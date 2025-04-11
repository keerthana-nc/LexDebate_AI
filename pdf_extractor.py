import requests
from bs4 import BeautifulSoup

class PDFExtractor:
    '''
    This class is responsible accessing all the html files accessed in web_scrapper class
    and extract useful information from the pdf files available in html files.
    '''

    def __init__(self, headers, base_url):
        # initialising required variables for scrapping website
        self.headers = headers
        self.base_url = base_url

    def get_pdf_links(self,html_links):
        '''
        This function is responsible for accessing pdf links from the html links
        :param html_files: A list of html links from the base url
        :return: A list of pdf links.
        '''

        pdf_links = []
        # to read content in html files
        for link in html_links:
            response = requests.get(link, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            for tag in soup.find_all('a', href=True):
                href = tag['href']
                if '.pdf' in href:
                    pdf_links.append(href)

        print(f"the number of pdf links obtained are:{len(pdf_links)}")  # 603 this number shld have been more than 1299

        return pdf_links

    def get_name(self):
        '''
        This function is responsible for extracting name of the case from each of the pdf link stored in pdf_links
        :return: A list of names containing the defense and prosecutor name of the case
        '''




