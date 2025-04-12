# app.py - Flask application for NJ Tax Court Cases with debugging (two years)
from flask import Flask, request, render_template_string
import re
import threading

# Importing classes which perform web scrapping
from new import WebScrapping
from pdf_extractor import PDFExtractor

app = Flask(__name__)
#Starting

# Base setup
BASE_URL = "https://law.justia.com/cases/new-jersey/tax-court/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

data_loaded = False  # set to true once the data has been loaded
error_message = None  # stores any error messages
loading_status = "Not started"  # current status is updated

# Store the scraped data
case_names = []
case_years = []
case_summaries = []
years_list = []


def load_data():
    """Load data by running scrapers with proper error handling"""
    global data_loaded, error_message, case_names, case_years, case_summaries, years_list, loading_status

    try:
        # Importing fitz from PyMuPDF for processing data from pdf file.
        import requests
        import fitz  # PyMuPDF

        # Initialize the classes
        loading_status = "Initializing scrapers..."
        scrapper = WebScrapping(headers=HEADERS, base_url=BASE_URL)
        extractor = PDFExtractor(headers=HEADERS, base_url=BASE_URL)

        # Get year-wise links
        loading_status = "Getting year links..."
        all_year_links = scrapper.get_year_links()
        year_links = all_year_links[:2]  # FIRST TWO YEARS
        print(f"Year links being processed: {year_links}")  # Debug info about years

        # Get HTML links
        loading_status = f"Getting HTML links from {len(year_links)} years..."
        html_links = scrapper.get_html_links(year_links)

        # Limiting HTML links per year for faster processing
        html_links_limited = []
        for year_index, year_link in enumerate(year_links):
            # Extract year from the link using regex
            year_match = re.search(r'/(\d{4})/?$', year_link)
            current_year = year_match.group(1) if year_match else "unknown"

            # Find HTML links for this specific year
            year_html_links = [link for link in html_links if year_link in link]
            # Take up to 20 links per year
            year_html_links = year_html_links[:30]
            html_links_limited.extend(year_html_links)
            print(f"Year {current_year}: Found {len(year_html_links)} HTML links")

        html_links = html_links_limited
        print(f"Total HTML links selected: {len(html_links)}")

        # Get PDF links
        loading_status = f"Extracting PDF links from {len(html_links)} HTML pages..."
        pdf_links = extractor.get_pdf_links(html_links)

        # Get all years from the PDF links
        available_years = set()
        for link in pdf_links:
            year_match = re.search(r'/(\d{4})-', link)
            if year_match:
                available_years.add(year_match.group(1))

        print(f"Available years in PDF links: {available_years}")

        # Balanced PDF selection from all available years
        balanced_pdf_links = []
        pdfs_per_year = 10  # Number of PDFs to take from each year

        for year in available_years:
            year_pdf_links = [link for link in pdf_links if f"/{year}-" in link][:pdfs_per_year]
            print(f"Selected {len(year_pdf_links)} PDFs from year {year}")
            balanced_pdf_links.extend(year_pdf_links)

        pdf_links = balanced_pdf_links

        loading_status = f"Found {len(pdf_links)} PDFs to process"
        print(f"DEBUG: Processing {len(pdf_links)} PDFs in total")

        # Extract content from PDFs
        loading_status = f"Extracting content from {len(pdf_links)} PDFs..."
        all_text_cases = []

        for i, pdf_url in enumerate(pdf_links):
            try:
                # Directly process each PDF
                loading_status = f"Processing PDF {i + 1} of {len(pdf_links)}: {pdf_url}"
                print(f"DEBUG: Processing PDF {i + 1}: {pdf_url}")

                response = requests.get(pdf_url, headers=HEADERS)
                if response.status_code == 200:
                    try:
                        doc = fitz.open(stream=response.content, filetype="pdf")
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        doc.close()
                        all_text_cases.append(text)
                        print(f"DEBUG: Successfully extracted text from PDF {i + 1}")
                    except Exception as e:
                        print(f"DEBUG: Error processing PDF {i + 1}: {str(e)}")
                        all_text_cases.append(f"Error processing PDF: {str(e)}")
                else:
                    print(f"DEBUG: HTTP error {response.status_code} for PDF {i + 1}")
                    all_text_cases.append(f"Error: Status code {response.status_code}")
            except Exception as e:
                print(f"DEBUG: Exception accessing PDF {i + 1}: {str(e)}")
                all_text_cases.append(f"Error accessing URL: {str(e)}")

        print(f"DEBUG: Extracted text from {len(all_text_cases)} PDFs")

        # Extract case names
        loading_status = "Extracting case names..."
        case_names = extractor.get_name(all_text_cases)
        print(f"DEBUG: Extracted {len(case_names)} case names")
        print(f"DEBUG: Case names: {case_names}")

        # Extract decision years
        loading_status = "Extracting case years..."
        case_years = extractor.find_year(all_text_cases)
        print(f"DEBUG: Extracted {len(case_years)} case years")
        print(f"DEBUG: Case years: {case_years}")

        # Extract summary
        loading_status = "Extracting case summaries..."
        case_summaries = extractor.find_summary(all_text_cases)
        print(f"DEBUG: Extracted {len(case_summaries)} case summaries")

        # Extract unique years for dropdown (using the actual years in the dataset)
        loading_status = "Preparing year filter..."
        years_set = set()
        for year_str in case_years:
            if 'year not found' not in year_str:
                match = re.search(r'\d{4}', year_str)
                if match:
                    years_set.add(match.group(0))

        # If no years were found, use years from PDF links
        if not years_set and available_years:
            years_set = available_years

        years_list = sorted(list(years_set), reverse=True)
        print(f"DEBUG: Unique years for dropdown: {years_list}")

        loading_status = f"Data loading complete. Found {len(case_names)} cases spanning {len(years_list)} years."
        data_loaded = True

    except Exception as e:
        error_message = f"Error loading data: {str(e)}"
        loading_status = "Error occurred during data loading."
        print(f"DEBUG: Fatal error in load_data: {str(e)}")

        # We'll consider data loaded even with errors
        data_loaded = True


# Start loading data in background thread
def start_data_loading():
    global loading_status
    loading_status = "Data loading started in background"
    thread = threading.Thread(target=load_data)
    thread.daemon = True
    thread.start()


# HTML template
MAIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>LexDebate AI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #333;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        select {
            padding: 8px;
            width: 200px;
        }
        button {
            padding: 8px 15px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .case {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .case-name {
            font-weight: bold;
            font-size: 18px;
            color: #2c3e50;
        }
        .case-year {
            color: #666;
            margin-bottom: 10px;
            font-style: italic;
        }
        .case-summary {
            margin-top: 10px;
            line-height: 1.5;
        }
        .error {
            background-color: #ffecec;
            color: #ff0000;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .info {
            background-color: #e7f3fe;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px;
        }
        .loading-bar {
            width: 100%;
            height: 20px;
            background-color: #f3f3f3;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .loading-bar-progress {
            height: 100%;
            background-color: #4CAF50;
            border-radius: 10px;
            width: 100%;
            animation: loading 2s infinite ease-in-out;
        }
        @keyframes loading {
            0% { width: 0%; }
            50% { width: 100%; }
            100% { width: 0%; }
        }
        .count-info {
            background-color: #f0f8ff;
            padding: 10px;
            margin-bottom: 15px;
            border-left: 3px solid #007bff;
        }
        .refresh-button {
            background-color: #007bff;
            margin-left: 10px;
        }
        .note {
            background-color: #fffde7;
            padding: 10px;
            border-left: 3px solid #fbc02d;
            margin-bottom: 15px;
        }
        .debug-info {
            background-color: #f2f2f2;
            padding: 10px;
            border: 1px solid #ddd;
            margin-top: 30px;
            font-family: monospace;
        }
    </style>
    <script>
        function reloadPage() {
            window.location.reload();
        }

        {% if not data_loaded %}
        // Auto refresh the page every 5 seconds until data is loaded
        setTimeout(reloadPage, 5000);
        {% endif %}
    </script>
</head>
<body>
    <div class="container">
        <h1>LexDebate AI</h1>

        <div class="note">
            <p><strong>Note:</strong> This application is processing cases from the two most recent years for demonstration.</p>
        </div>

        {% if error_message %}
        <div class="error">
            <p><strong>Warning:</strong> {{ error_message }}</p>
            {% if not case_names or case_names|length < 3 %}
            <p>Showing sample data for demonstration purposes.</p>
            {% else %}
            <p>Displaying partial data that was successfully loaded.</p>
            {% endif %}
        </div>
        {% endif %}

        {% if not data_loaded %}
        <div class="loading">
            <h3>Loading Case Data</h3>
            <p>{{ loading_status }}</p>
            <div class="loading-bar">
                <div class="loading-bar-progress"></div>
            </div>
            <p>This process may take a few minutes. The page will refresh automatically.</p>
        </div>
        {% else %}

        <div class="count-info">
            <p><strong>Database Status:</strong> {{ loading_status }}</p>
        </div>

        <div class="form-group">
            <form method="get">
                <label for="year">Select Year to Filter Cases:</label>
                <select name="year" id="year">
                    <option value="">All Years</option>
                    {% for year in years %}
                        <option value="{{ year }}" {% if year == selected_year %}selected{% endif %}>{{ year }}</option>
                    {% endfor %}
                </select>
                <button type="submit">Filter Cases</button>
                <button type="button" class="refresh-button" onclick="reloadPage()">Refresh Data</button>
            </form>
        </div>

        {% if cases %}
            <h2>Cases from {{ selected_year if selected_year else "All Years" }}</h2>
            <p>Found {{ cases|length }} case(s)</p>

            {% for case in cases %}
                <div class="case">
                    <div class="case-name">{{ case.name }}</div>
                    <div class="case-year">{{ case.year }}</div>
                    <div class="case-summary">
                        <strong>Summary:</strong> {{ case.summary[:500] }}...
                    </div>
                </div>
            {% endfor %}
        {% else %}
            <p>No cases found for the selected criteria.</p>
        {% endif %}

        <!-- Debug Information -->
        <div class="debug-info">
            <h3>Debug Information</h3>
            <p><strong>Data Loaded:</strong> {{ data_loaded }}</p>
            <p><strong>Years Available:</strong> {{ years }}</p>
            <p><strong>Selected Year:</strong> "{{ selected_year }}"</p>
            <p><strong>Case Names ({{ case_names|length }}):</strong> {{ case_names }}</p>
            <p><strong>Case Years ({{ case_years|length }}):</strong> {{ case_years }}</p>
            <p><strong>Case Summaries ({{ case_summaries|length }}):</strong> [summaries available]</p>
            <p><strong>Cases Found:</strong> {{ cases|length }}</p>
        </div>

        {% endif %}
    </div>
</body>
</html>
'''


@app.route('/')
def index():
    global data_loaded

    # Start data loading on first request if not already started
    if not data_loaded and loading_status == "Not started":
        start_data_loading()

    # Get the selected year from the query parameters
    selected_year = request.args.get('year', '')
    print(f"DEBUG: Selected year from query: '{selected_year}'")

    # Filter cases based on the selected year
    filtered_cases = []
    if data_loaded:
        print(f"DEBUG: Filtering cases for year: '{selected_year}'")
        for i in range(len(case_names)):
            if i < len(case_years) and i < len(case_summaries):
                # If no year is selected or the year matches
                year_match = not selected_year or selected_year in case_years[i]
                print(f"DEBUG: Case {i}: {case_names[i]} - Year: {case_years[i]} - Match: {year_match}")
                if year_match:
                    filtered_cases.append({
                        'name': case_names[i],
                        'year': case_years[i],
                        'summary': case_summaries[i]
                    })
        print(f"DEBUG: Found {len(filtered_cases)} matching cases")

    # Render the template with the data
    return render_template_string(
        MAIN_TEMPLATE,
        years=years_list,
        selected_year=selected_year,
        cases=filtered_cases,
        data_loaded=data_loaded,
        error_message=error_message,
        loading_status=loading_status,
        case_names=case_names,
        case_years=case_years,
        case_summaries=case_summaries
    )


if __name__ == '__main__':
    app.run(debug=True)