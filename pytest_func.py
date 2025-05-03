import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from bs4 import BeautifulSoup
import fitz
from pdf_extractor import PDFExtractor
from new import WebScrapping


# Test fixtures
@pytest.fixture
def mock_headers():
    return {"User-Agent": "Mozilla/5.0"}


@pytest.fixture
def mock_base_url():
    return "https://law.justia.com/cases/new-jersey/tax-court/"


@pytest.fixture
def web_scrapper(mock_headers, mock_base_url):
    return WebScrapping(mock_headers, mock_base_url)


@pytest.fixture
def pdf_extractor(mock_headers, mock_base_url):
    return PDFExtractor(mock_headers, mock_base_url)


# Test 1: Test get_year_links functionality
@patch('requests.get')
def test_get_year_links(mock_get, web_scrapper):
    """Test that get_year_links correctly extracts year links from the base URL"""
    # Mock HTML response containing year links
    mock_html = """
    <html>
        <body>
            <a href="/cases/new-jersey/tax-court/2025/">2025</a>
            <a href="/cases/new-jersey/tax-court/2024/">2024</a>
            <a href="/cases/new-jersey/tax-court/2023/">2023</a>
            <a href="/some/other/link.html">Other Link</a>
        </body>
    </html>
    """
    mock_response = Mock()
    mock_response.text = mock_html
    mock_get.return_value = mock_response

    year_links = web_scrapper.get_year_links()

    # Verify only year links are extracted and HTML links are filtered out
    assert len(year_links) == 2  # Limited to 2 for demo
    assert "https://law.justia.com/cases/new-jersey/tax-court/2025/" in year_links[0]
    assert "https://law.justia.com/cases/new-jersey/tax-court/2024/" in year_links[1]


# Test 2: Test get_pdf_links functionality
@patch('requests.get')
def test_get_pdf_links(mock_get, pdf_extractor):
    """Test that get_pdf_links correctly extracts PDF links from HTML pages"""
    # Mock HTML response containing PDF links
    mock_html = """
    <html>
        <body>
            <a href="https://cases.justia.com/new-jersey/tax-court/2025-006902-2020.pdf?ts=1745423153">Document 1</a>
            <a href="https://cases.justia.com/new-jersey/tax-court/2025-008302-2022.pdf?ts=1743608133">Document 2</a>
        </body>
    </html>
    """
    mock_response = Mock()
    mock_response.text = mock_html
    mock_get.return_value = mock_response

    html_links = ["https://law.justia.com/cases/new-jersey/tax-court/2025/006902-2020.html", "https://law.justia.com/cases/new-jersey/tax-court/2025/008302-2022.html"]
    pdf_links = pdf_extractor.get_pdf_links(html_links)

    # Verify PDF links are extracted correctly
    assert len(pdf_links) <= 3  # Limited to 3 for faster execution
    assert all('.pdf' in link for link in pdf_links)


# Test 3: Test get_name functionality for case name extraction
def test_get_name(pdf_extractor):
    """Test that get_name correctly extracts case names from PDF content"""
    # Mock case content with proper formatting
    all_text_cases = [
        """NOT FOR PUBLICATION WITHOUT APPROVAL OF THE TAX COURT COMMITTEE ON OPINIONS
        JOHN DOE,
        : TAX COURT OF NEW JERSEY
        Plaintiff,                : DOCKET NO. 123456-2024
        v.                     :
        STATE OF NEW JERSEY,   :
        Defendant.             :""",

        "JANE SMITH v. CITY OF NEWARK",

        "Invalid content without proper case name format"
    ]

    case_names = pdf_extractor.get_name(all_text_cases)

    # Verify case names are extracted correctly
    assert len(case_names) >= 2
    assert "JOHN DOE v. STATE OF NEW JERSEY" in case_names
    assert "JANE SMITH v. CITY OF NEWARK" in case_names


# Test 4: Test find_year functionality
def test_find_year(pdf_extractor):
    """Test that find_year correctly extracts decision dates from case content"""
    all_text_cases = [
        "This case was Decided March 15, 2024. The court finds...",
        "IN THE TAX COURT OF NEW JERSEY\nDecided: July 20, 2023\nJudgment...",
        "Before the court on November 10, 2024. This matter...",
        "No date information in this case"
    ]

    case_years = pdf_extractor.find_year(all_text_cases)

    # Verify years are extracted correctly
    assert len(case_years) == 4
    assert "March 15, 2024" in case_years
    assert "July 20, 2023" in case_years
    assert "November 10, 2024" in case_years
    assert "year not found" in case_years


# Test 5: Test get_content functionality with PDF processing
@patch('requests.get')
@patch('fitz.open')
def test_get_content(mock_fitz_open, mock_get, pdf_extractor):
    """Test that get_content correctly extracts text from PDF files"""
    # Mock PDF response
    mock_pdf_response = Mock()
    mock_pdf_response.content = b'%PDF-1.4...'  # Mock PDF header
    mock_pdf_response.headers = {'Content-Type': 'application/pdf'}
    mock_get.return_value = mock_pdf_response

    # Mock PyMuPDF document
    mock_page = Mock()
    mock_page.get_text.return_value = "Sample PDF content"
    mock_doc = Mock()
    mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
    mock_doc.close = Mock()
    mock_fitz_open.return_value = mock_doc

    pdf_links = ["https://cases.justia.com/new-jersey/tax-court/2025-006902-2020.pdf?ts=1745423153", "https://cases.justia.com/new-jersey/tax-court/2025-008302-2022.pdf?ts=1743608133"]
    all_text_cases = pdf_extractor.get_content(pdf_links)

    # Verify content is extracted correctly
    assert len(all_text_cases) == 2
    assert "Sample PDF content" in all_text_cases[0]
    assert mock_doc.close.called