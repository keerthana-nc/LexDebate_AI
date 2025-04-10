import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF

# Step 1: Load base URL
base_url = "https://law.justia.com/cases/new-jersey/tax-court/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Step 2: Get HTML case links
response = requests.get(base_url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

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

