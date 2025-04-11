from new import WebScrapping
from pdf_extractor import PDFExtractor
import requests

# Base setup
BASE_URL = "https://law.justia.com/cases/new-jersey/tax-court/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

def main():
    # Step 1: Initialize the classes
    scrapper = WebScrapping(headers=HEADERS, base_url=BASE_URL)
    extractor = PDFExtractor(headers=HEADERS, base_url=BASE_URL)

    # Step 2: Get year-wise links
    year_links = scrapper.get_year_links()

    # Step 3: Get all case HTML links from year links
    html_links = scrapper.get_html_links(year_links)

    # Step 4: Extract all PDF links from those HTML files
    pdf_links = extractor.get_pdf_links(html_links)

    # Step 5: Extract content from each PDF
    all_text_cases = extractor.get_content(pdf_links)

    # Step 6: Extract case names
    case_names = extractor.get_name(all_text_cases)

    # Step 7: Extract decision years
    case_years = extractor.find_year(all_text_cases)

    # Step 8: Extract summary (last 1500 characters)
    case_summaries = extractor.find_summary(all_text_cases)

    # Step 9: Print a few results
    print("\nSample Case Details:\n")
    for i in range(min(5, len(case_names))):
        print(f"Case {i+1}:")
        print(f"Name     : {case_names[i]}")
        print(f"Decided On: {case_years[i]}")
        print(f"Summary  : {case_summaries[i][:300]}...\n")  # Displaying only the first 300 characters of the summary

    print(f"Total cases processed: {len(case_names)}")

if __name__ == "__main__":
    main()
