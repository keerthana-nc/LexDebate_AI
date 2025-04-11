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
                year_link = urljoin(base_url, href)
                # year_link contains https://law.justia.com/cases/new-jersey/tax-court/2025
                year_links.append(year_link)
        # year_links list has all the year links along with only 15 html links from 2025 and 2024 combined.
        # Removing html links from year_links to have only year links then access html links from all these links.
        filtered_year_links = []
        for link in year_links:
            if not link.endswith('html'):
                filtered_year_links.append(link)
        year_links = filtered_year_links

        print(f"the length of year_links after removing html files are:{len(year_links)}")

        return year_links

    def get_html_links(self):
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

        print(f"the total number of html files read are: {len(html_links)}")  # 1299

        return html_links









# Step 1: Load base URL
base_url = "https://law.justia.com/cases/new-jersey/tax-court/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Step 2: Get HTML case links
response = requests.get(base_url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# base url links -> year links -> a list of html links -> pdf document opens
# Step 3: Get year links
import re
year_links = [] # list that contains links of all the years available in base url.

for tag in soup.find_all('a', href = True):
    href = tag['href']
    if re.match(r'/cases/new-jersey/tax-court/\d{4}/', href):
        # href contains part of the url eg: a -> <a href="/cases/new-jersey/tax-court/2025/">2025</a> ;
        # href: /cases/new-jersey/tax-court/2025/
        year_link = urljoin(base_url, href)
        # year_link contains https://law.justia.com/cases/new-jersey/tax-court/2025
        year_links.append(year_link)
#year_links list has all the year links along with only 15 html links from 2025 and 2024 combined.
# Removing html links from year_links to have only year links then access html links from all these links.
filtered_year_links = []
for link in year_links:
    if not link.endswith('html'):
        filtered_year_links.append(link)
year_links = filtered_year_links

print(f"the length of year_links after removing html files are:{len(year_links)}") # 44

# Now that we have year_links we need to access html files from those links
html_links = []
for link in year_links:
    response = requests.get(link, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    for tag in soup.find_all('a', href=True):
        href_links = tag['href'] # href_links contains hyperlinks from those links which contain href -> href - True
        if href_links.endswith('html'):
            html_link  = urljoin(link, href_links)
            html_links.append(html_link)

print(f"the total number of html files read are: {len(html_links)}") #1299

# Now that html files are read, these html files contain pdf documents.
# next step is to read those pdf documents.

pdf_links = []
# to read content in html files
for link in html_links:
    response = requests.get(link, headers = headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    for tag in soup.find_all('a', href = True):
        href = tag['href']
        if '.pdf' in href:
            pdf_links.append(href)

print(f"the number of pdf links obtained are:{len(pdf_links)}") #603 this number shld have been more than 1299











'''
case_links = []
for tag in soup.find_all('a', href=True):
    href = tag['href']
    if href.endswith('.html'):
        full_link = urljoin(base_url, href)
        case_links.append(full_link)

print(f"Number of html files accessed are:",len(case_links))

all_text_cases = []

# Step 3: Visit each case page and download PDF
for case_url in case_links:
    case_response = requests.get(case_url, headers=headers)
    case_soup = BeautifulSoup(case_response.text, 'html.parser')

    pdf_tag = case_soup.find('a', string='Download PDF')
    if pdf_tag:
        pdf_url = urljoin(case_url, pdf_tag['href'])
        print(f"Accessed pdf links {pdf_url}")
        pdf_response = requests.get(pdf_url, headers=headers)

        try:
            doc = fitz.open(stream=pdf_response.content, filetype="pdf")
            all_text = ""
            for page in doc:
                all_text += page.get_text()
            doc.close()

            print("Accessed content in the pdf file\n")
            #print(all_text[:1000])  # Preview first 1000 characters

            all_text_cases.append(all_text)


        except Exception as e:
            print(f" Failed to process PDF: {e}")
    else:
        print(f" No PDF found at: {case_url}")

#print(f"all_text_cases is a list where each element must contain data of each pdf file:{all_text_cases}")

#regular expression -re used for matching strings , patterns.
import re

for i in range(len(all_text_cases)):
    # Clean text
    case_text = all_text_cases[i].strip() # removes whitespaces before and after the content in list.
    case_text = re.sub(r'\s+', ' ', case_text) # removes tabs or next lines or any kind of
    # space that is represented by s\+ and is replaced by a single space.
    all_text_cases[i] = case_text #updates the list

def find_name(all_text_cases):
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

    return case_names

def find_year(all_text_cases):

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

    return case_years

def find_summary(all_text_cases):
    summary = []
    # deciding to extract the last
    for case in all_text_cases:
        summaries = case[-1500:]
        #if i strip the spaces in between the words would it make it lose its meaning to the model?
        #summaries.strip(" ")
        summary.append(summaries)

    return summary

# Calling the function
case_names = find_name(all_text_cases)
print(case_names)  # This will print all case names found
case_years = find_year(all_text_cases)
print(case_years)
summary = find_summary(all_text_cases)
print(summary)
'''
