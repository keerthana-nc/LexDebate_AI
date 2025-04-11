import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF
import re


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

    def get_content(self, pdf_links):
        '''
        This function is responsible for extracting content of the case from each of the pdf link stored in pdf_links
        :return: A list that contains the text extracted from the pdf links
        '''

        all_text_cases = []

        for case_url in pdf_links:
            case_response = requests.get(case_url, headers=self.headers)
            case_soup = BeautifulSoup(case_response.text, 'html.parser')

            pdf_tag = case_soup.find('a', string='Download PDF')
            if pdf_tag:
                pdf_url = urljoin(case_url, pdf_tag['href'])
                print(f"Accessed pdf links {pdf_url}")
                pdf_response = requests.get(pdf_url, headers=self.headers)

                try:
                    doc = fitz.open(stream=pdf_response.content, filetype="pdf")
                    all_text = ""
                    for page in doc:
                        all_text += page.get_text()
                    doc.close()

                    print("Accessed content in the pdf file\n")
                    # print(all_text[:1000])  # Preview first 1000 characters

                    all_text_cases.append(all_text)


                except Exception as e:
                    print(f" Failed to process PDF: {e}")
            else:
                print(f" No PDF found at: {case_url}")

        print(f"all_text_cases is a list where each element must contain data of each pdf file and the length of all_text_cases is:{len(all_text_cases)}")

        return all_text_cases

    def get_name(self, all_text_cases):
        '''
        Extracts case names from the case file content.
        :param all_text_cases: a list that contains the text in each case file
        :return: returns a list that contains only the case names.
        '''
        for i in range(len(all_text_cases)):
            # Clean text
            case_text = all_text_cases[i].strip()  # removes whitespaces before and after the content in list.
            case_text = re.sub(r'\s+', ' ', case_text)  # removes tabs or next lines or any kind of
            # space that is represented by s\+ and is replaced by a single space.
            all_text_cases[i] = case_text  # updates the list

        case_names = []
        avoid = "NOT FOR PUBLICATION WITHOUT APPROVAL OF THE TAX COURT COMMITTEE ON OPINIONS"

        for case in all_text_cases:
            pattern = r'([A-Z][A-Z\s,\.]+),\s*:\s*.*?Plaintiff,\s*.*?v\.\s*.*?([A-Z][A-Z\s,\.]+),\s*.*?Defendant\.'
            match = re.search(pattern, case, re.DOTALL)

            if match:
                plaintiff = match.group(1).strip()
                defendant = match.group(2).strip()
                if plaintiff.startswith(avoid):
                    plaintiff = plaintiff.replace(avoid, "").strip()
                case_names.append(f"{plaintiff} v. {defendant}")
            else:
                # Another approach - check title or header sections
                header_pattern = r'((?:[A-Z][A-Z\s,\.]+){1,5})\s+v\.\s+((?:[A-Z][A-Z\s,\.]+){1,5})'
                lines = case.split('\n')[:70]  # Check first 20 lines

                for line in lines:
                    header_match = re.search(header_pattern, line)
                    if header_match:
                        plaintiff = header_match.group(1).strip()
                        defendant = header_match.group(2).strip()

                        # Remove the unwanted text if present
                        if avoid in plaintiff:
                            plaintiff = plaintiff.replace(avoid, "").strip()
                        if avoid in defendant:
                            defendant = defendant.replace(avoid, "").strip()

                        case_names.append(f"{plaintiff} v. {defendant}")
        print(f"The number of case names extracted is: {len(case_names)}")
        return case_names

    def find_year(self,all_text_cases):
        '''
        This function extracts the year the case was processed in from the content extracted from pdf.
        :param all_text_cases: a list that contains the text in each case file
        :return: A list of years of case proceedings.
        '''

        case_years = []

        for case in all_text_cases:
            # the date and year comes after the word decided in the first page of the pdf
            decided_match = re.search(r'Decided\s+([A-Za-z]+\s+\d+,\s+\d{4})', case)
            decided_match2 = re.search(r'Decided:\s+([A-Za-z]+\s+\d+,\s+\d{4})', case)
            decided_match3 = re.search(r'([A-Za-z]+\s+\d+,\s+\d{4})', case[:1000])
            if decided_match:
                year_date = decided_match.group(1)
                case_years.append(year_date)
            elif decided_match2:
                year_date = decided_match2.group(1)
                case_years.append(year_date)
            elif decided_match3:
                year_date = decided_match3.group(1)
                case_years.append(year_date)
            else:
                case_years.append("year not found")
        print(f"The number of years extracted is :{len(case_years)}")

        return case_years

    def find_summary(self, all_text_cases):
        '''
        This function extracts the last 1500 characters from each case content, indicating summary of the entire case proceedings.
        :param all_text_cases: a list that contains the text in each case file
        :return: A list that contains the last 1500 characters for each case as each element.
        '''
        summary = []
        # deciding to extract the last
        for case in all_text_cases:
            summaries = case[-1500:]
            summary.append(summaries)

        return summary




