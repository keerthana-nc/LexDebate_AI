from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
import threading
import time
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF (pip install PyMuPDF)

# Import the module
import prepping_JSON

# Creating objects for the class to use the functions in these classes later.
case_db = prepping_JSON.CaseDatabase()
case_db.load_database()  # Load existing database
case_embeddings = prepping_JSON.CaseEmbeddings()
case_embeddings.load_embeddings()  # Load existing embeddings

# Create the RAG instance
legal_rag = prepping_JSON.LegalRAG(case_db, case_embeddings)

# Create Flask app
app = Flask(__name__)

# Base setup
BASE_URL = "https://law.justia.com/cases/new-jersey/tax-court/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

# Global variables to store database status
database_status = {
    "building": False,
    "message": "",
    "progress": 0
}

# Setup directories
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
database_path = data_dir / "case_database.json"
embeddings_path = data_dir / "embeddings.json"


# Classes for the LexDebate AI system
class WebScrapping:
    '''
    This class is responsible for scrapping the tax court website to get links to cases
    '''

    def __init__(self, headers, base_url):
        self.headers = headers
        self.base_url = base_url

    def get_year_links(self):
        '''
        Gets links to all years available on the website
        '''
        response = requests.get(self.base_url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        year_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if "/cases/new-jersey/tax-court/" in href and href.endswith('/'):
                if any(year in href for year in ['2020', '2019']):  # Limit to recent years
                    year_links.append(href)

        return year_links[:2]  # Limit to the most recent 2 years for demo

    def get_html_links(self, year_links):
        '''
        Gets links to all case HTML pages for each year
        '''
        html_links = []

        for year_link in year_links:
            response = requests.get(year_link, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if "/cases/new-jersey/tax-court/" in href and href.endswith('.html'):
                    html_links.append(href)

        return html_links[:10]  # Limit to 10 HTML pages for demo


class PDFExtractor:
    '''
    This class is responsible for extracting content from PDFs
    '''

    def __init__(self, headers, base_url):
        self.headers = headers
        self.base_url = base_url

    def get_pdf_links(self, html_links):
        '''
        Gets links to PDFs from HTML pages
        '''
        pdf_links = []

        for link in html_links:
            response = requests.get(link, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            for tag in soup.find_all('a', href=True):
                href = tag['href']
                if '.pdf' in href:
                    full_url = urljoin(link, href)
                    pdf_links.append(full_url)

        return pdf_links[:5]  # Limit to 5 PDFs for demo

    def get_content(self, pdf_links):
        '''
        Extracts content from PDFs
        '''
        all_text_cases = []

        for pdf_url in pdf_links:
            try:
                print(f"Accessing PDF: {pdf_url}")
                pdf_response = requests.get(pdf_url, headers=self.headers)

                if pdf_response.headers.get('Content-Type', '').lower() == 'application/pdf' or pdf_response.content[
                                                                                                :4] == b'%PDF':
                    try:
                        doc = fitz.open(stream=pdf_response.content, filetype="pdf")
                        all_text = ""
                        for page in doc:
                            all_text += page.get_text()
                        doc.close()

                        all_text_cases.append(all_text)
                    except Exception as e:
                        all_text_cases.append("Failed to extract text")
                else:
                    all_text_cases.append("Not a valid PDF")
            except Exception as e:
                print(f"Error accessing URL {pdf_url}: {e}")
                all_text_cases.append("Error accessing URL")

        return all_text_cases

    def get_name(self, all_text_cases):
        '''
        Extracts case names from content
        '''
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
                # Try another approach
                header_pattern = r'((?:[A-Z][A-Z\s,\.]+){1,5})\s+v\.\s+((?:[A-Z][A-Z\s,\.]+){1,5})'
                lines = case.split('\n')[:70]

                for line in lines:
                    header_match = re.search(header_pattern, line)
                    if header_match:
                        plaintiff = header_match.group(1).strip()
                        defendant = header_match.group(2).strip()

                        if avoid in plaintiff:
                            plaintiff = plaintiff.replace(avoid, "").strip()
                        if avoid in defendant:
                            defendant = defendant.replace(avoid, "").strip()

                        case_names.append(f"{plaintiff} v. {defendant}")
                        break
                else:
                    case_names.append("Unknown Case")

        return case_names

    def find_year(self, all_text_cases):
        '''
        Extracts case years
        '''
        case_years = []

        for case in all_text_cases:
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
                case_years.append("Year not found")

        return case_years


class LegalCase:
    def __init__(self, case_id, name, year, text):
        self.case_id = case_id
        self.name = name
        self.year = year
        self.text = text[:50000]  # Limit to 50,000 characters

    def to_dict(self):
        return {
            "name": self.name,
            "year": self.year,
            "Text": self.text
        }

    def __str__(self):
        return f"{self.name} ({self.year})"


class CaseDatabase:
    def __init__(self):
        self.cases = {}
        self.database_path = database_path

    def create_database(self):
        global database_status

        try:
            database_status["building"] = True
            database_status["message"] = "Starting to retrieve information..."
            database_status["progress"] = 5

            # Initialize scraping tools
            scrapper = WebScrapping(headers=HEADERS, base_url=BASE_URL)
            extractor = PDFExtractor(headers=HEADERS, base_url=BASE_URL)

            # Step 2: Get year-wise links
            database_status["message"] = "Getting year links..."
            database_status["progress"] = 10
            year_links = scrapper.get_year_links()

            # Step 3: Get HTML links from years
            database_status["message"] = "Getting HTML links..."
            database_status["progress"] = 20
            html_links = scrapper.get_html_links(year_links)

            # Step 4: Get PDF links from HTML pages
            database_status["message"] = "Getting PDF links..."
            database_status["progress"] = 30
            pdf_links = extractor.get_pdf_links(html_links)

            # Step 5: Extract content from PDFs
            database_status["message"] = "Extracting content from PDFs..."
            database_status["progress"] = 40
            all_text_cases = extractor.get_content(pdf_links)

            # Step 6: Extract case names
            database_status["message"] = "Extracting case names..."
            database_status["progress"] = 60
            case_names = extractor.get_name(all_text_cases)

            # Step 7: Extract decision years
            database_status["message"] = "Extracting decision years..."
            database_status["progress"] = 70
            case_years = extractor.find_year(all_text_cases)

            # Create cases dictionary
            database_status["message"] = "Building case database..."
            database_status["progress"] = 80
            self.cases = {}

            for i, (name, year, content) in enumerate(zip(case_names, case_years, all_text_cases)):
                case_id = f"case_{i}"
                self.cases[case_id] = LegalCase(case_id, name, year, content)

            # Save database
            database_status["message"] = "Saving database..."
            database_status["progress"] = 90
            self.save_database()

            database_status["message"] = f"Database created with {len(self.cases)} cases"
            database_status["progress"] = 100
            database_status["building"] = False

            return self.cases

        except Exception as e:
            database_status["message"] = f"Error: {str(e)}"
            database_status["building"] = False
            database_status["progress"] = 0
            return {}

    def save_database(self):
        cases_dict = {case_id: case.to_dict() for case_id, case in self.cases.items()}

        with open(self.database_path, 'w', encoding='utf-8') as f:
            json.dump(cases_dict, f, ensure_ascii=False, indent=2)

    def load_database(self):
        if self.database_path.exists():
            with open(self.database_path, 'r', encoding='utf-8') as f:
                cases_dict = json.load(f)

            self.cases = {}
            for case_id, case_data in cases_dict.items():
                self.cases[case_id] = LegalCase(
                    case_id=case_id,
                    name=case_data["name"],
                    year=case_data["year"],
                    text=case_data["Text"]
                )

            return self.cases
        else:
            return {}

    def get_case(self, case_id):
        return self.cases.get(case_id)


class CaseEmbeddings:
    def __init__(self):
        self.embeddings = {}
        self.embeddings_path = embeddings_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_embeddings(self, cases):
        global database_status

        try:
            database_status["message"] = "Creating embeddings..."
            database_status["progress"] = 80

            for case_id, case in cases.items():
                text = case.text[:5000]
                embedding = self.model.encode(text)
                self.embeddings[case_id] = embedding

            database_status["message"] = "Saving embeddings..."
            database_status["progress"] = 90
            self.save_embeddings()

            database_status["message"] = f"Embeddings created for {len(self.embeddings)} cases"
            database_status["progress"] = 100

            return self.embeddings

        except Exception as e:
            database_status["message"] = f"Error creating embeddings: {str(e)}"
            return {}

    def save_embeddings(self):
        serializable_embeddings = {k: v.tolist() for k, v in self.embeddings.items()}

        with open(self.embeddings_path, 'w') as f:
            json.dump(serializable_embeddings, f)

    def load_embeddings(self):
        if self.embeddings_path.exists():
            with open(self.embeddings_path, 'r') as f:
                serialized_embeddings = json.load(f)

            import numpy as np
            self.embeddings = {k: np.array(v) for k, v in serialized_embeddings.items()}

            return self.embeddings
        else:
            return {}

    def encode_query(self, query):
        return self.model.encode(query)


class LegalRAG:
    def __init__(self, case_database, case_embeddings):
        # These member variables are essential for the RAG system as they provide
        # access to the cases, their embeddings, and the LLM needed for response generation
        self.case_database = case_database
        self.case_embeddings = case_embeddings
        self.tokenizer = None
        self.model = None
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import re  # Import regex for section extraction
        self.re = re

    def find_relevant_cases(self, query, top_n=3):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        similarities = {}  # a dict that holds case id as the key and a number that says how similar is that case with the query as the value

        # embed the query
        query_embedding = self.case_embeddings.encode_query(query)

        # Make sure query embedding is a numpy array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)

        # Reshape for sklearn's cosine_similarity function
        query_embedding_reshaped = query_embedding.reshape(1, -1)

        # after embedding the user query, find relevant cases from case_embeddings / embeddings -> dict
        # to find relevant cases from embeddings dictionary related to query
        for case_id, case_embedding in self.case_embeddings.embeddings.items():
            # Make sure case embedding is a numpy array
            if not isinstance(case_embedding, np.ndarray):
                case_embedding = np.array(case_embedding)

            # Reshape for sklearn's function
            case_embedding_reshaped = case_embedding.reshape(1, -1)

            # Calculate similarity
            similarity = cosine_similarity(case_embedding_reshaped, query_embedding_reshaped)[0][0]

            # Store similarity score
            similarities[case_id] = similarity

        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # top 3 of the sorted similarities
        top_cases = sorted_similarities[:top_n]
        # top_cases is a list of tuples,
        # [(case_id1, similarity_score1), (case_id2, similarity_score2), (case_id3, similarity_score3)]

        relevant_cases = []  # this is to hold the content (non embedded form) of cases relevant to the query
        for case_id, similarity in top_cases:
            # Now case_id contains the ID (e.g., "case_0")
            # and similarity contains the similarity score (e.g., 0.85)
            case = self.case_database.get_case(case_id)
            relevant_cases.append({
                'case_id': case_id,
                'name': case.name,
                'year': case.year,
                'similarity': similarity
            })

        return relevant_cases
    def generate_response(self, query, mode, relevant_cases):
        try:
            if self.tokenizer is None or self.model is None:
                print("Loading Flan-T5 model...")
                # Use a larger model for better quality responses
                model_name = "google/flan-t5-large"  # Upgrade from base to large
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
                print(f"Successfully loaded {model_name} model")

            # Extract key sections from cases for better context
            structured_context = ""
            for i, case_data in enumerate(relevant_cases):
                case_id = case_data['case_id']
                case = self.case_database.get_case(case_id)
                case_text = case.text

                # Try to extract key sections using regex
                facts_section = ""
                issue_section = ""
                holding_section = ""

                # Extract facts section
                facts_match = self.re.search(
                    r'(?i)FACTS?:|STATEMENT\s+OF\s+FACTS?:|BACKGROUND:(.*?)(?:ISSUE|HOLDING|OPINION|DISCUSSION|CONCLUSION)',
                    case_text, self.re.DOTALL)
                if facts_match:
                    facts_section = facts_match.group(1).strip()[:1000]  # Limit to 1000 chars

                # Extract issue section
                issue_match = self.re.search(
                    r'(?i)ISSUE?:|QUESTION\s+PRESENTED:(.*?)(?:HOLDING|OPINION|DISCUSSION|CONCLUSION)',
                    case_text, self.re.DOTALL)
                if issue_match:
                    issue_section = issue_match.group(1).strip()[:500]  # Limit to 500 chars

                # Extract holding section
                holding_match = self.re.search(r'(?i)HOLDING:|CONCLUSION:(.*?)(?:ORDER|$)', case_text, self.re.DOTALL)
                if holding_match:
                    holding_section = holding_match.group(1).strip()[:1000]  # Limit to 1000 chars

                # Build structured context for this case
                structured_context += f"Case {i + 1}: {case.name} ({case.year})\n"

                if facts_section:
                    structured_context += f"FACTS: {facts_section}\n\n"
                if issue_section:
                    structured_context += f"ISSUE: {issue_section}\n\n"
                if holding_section:
                    structured_context += f"HOLDING: {holding_section}\n\n"

                # If we couldn't extract structured sections, use a segment of the full text
                if not (facts_section or issue_section or holding_section):
                    # Take a larger chunk, but still manageable
                    structured_context += f"EXCERPT: {case_text[:1500]}\n\n"

                structured_context += "-" * 50 + "\n\n"

            # Create detailed prompts based on mode
            if mode == "1":  # Advice
                instruction = (
                    f"You are a tax law specialist providing legal advice based on tax court cases. "
                    f"A client has asked: '{query}'\n\n"
                    f"Structure your response with these sections:\n"
                    f"1. ISSUE: Identify the specific legal issue in the query.\n"
                    f"2. RELEVANT LAW: Cite the relevant principles from the tax court cases.\n"
                    f"3. ANALYSIS: Apply these principles to the client's situation.\n"
                    f"4. ADVICE: Provide specific steps the client should take, including deadlines, "
                    f"potential penalties, and available options.\n\n"
                    f"Be specific, practical, and cite the relevant cases in your analysis."
                )
            elif mode == "2":  # Arguments
                instruction = (
                    f"You are a tax law attorney preparing arguments for a case involving: '{query}'\n\n"
                    f"Structure your response with these sections:\n"
                    f"1. KEY FACTS: Summarize the essential facts of the case.\n"
                    f"2. LEGAL ISSUES: Identify the central legal questions.\n"
                    f"3. PLAINTIFF'S ARGUMENTS: Present the strongest arguments for the plaintiff, "
                    f"with supporting case law and reasoning.\n"
                    f"4. DEFENDANT'S ARGUMENTS: Present the strongest arguments for the defendant, "
                    f"with supporting case law and reasoning.\n"
                    f"5. LIKELY OUTCOME: Based on precedent, analyze the potential ruling.\n\n"
                    f"Make detailed, specific legal arguments citing the relevant cases. "
                    f"Focus on tax law principles and precedents."
                )
            else:  # General Response
                instruction = (
                    f"You are a tax law expert responding to this query: '{query}'\n\n"
                    f"Using the tax court cases provided, give a comprehensive answer that:\n"
                    f"1. Addresses the specific question asked\n"
                    f"2. Explains relevant tax law principles\n"
                    f"3. Cites specific cases and their holdings\n"
                    f"4. Provides context and implications\n\n"
                    f"Make your response detailed, accurate, and focused on the query."
                )

            # Combine instruction and context
            input_text = f"{instruction}\n\nRELEVANT CASE PRECEDENTS:\n\n{structured_context}\n\nDETAILED RESPONSE:"

            # Generate response with improved parameters
            inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024).to(self.device)

            # Use better generation parameters
            outputs = self.model.generate(
                **inputs,
                max_length=768,  # Allow longer responses
                num_beams=5,  # More beam search paths
                temperature=0.7,  # Add some creativity
                do_sample=True,  # Enable sampling
                top_p=0.9,  # Nucleus sampling
                early_stopping=True
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Check if response is just echoing input or too short
            if response == query or len(response) < 50 or response == input_text:
                print("Model generated an inadequate response, falling back to template")
                # Fallback to template-based response
                if mode == "1":  # Advice
                    result = self.generate_template_advice(query, relevant_cases)
                elif mode == "2":  # Arguments
                    result = self.generate_template_arguments(query, relevant_cases)
                else:  # General
                    result = self.generate_template_general(query, relevant_cases)
                return result

            return {
                "response": response,
                "relevant_cases": relevant_cases
            }
        except Exception as e:
            print(f"Error generating response with LLM: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to rule-based response generation
            return self.generate_simple_response(query, mode, relevant_cases)

    def generate_template_advice(self, query, relevant_cases):
        """Generate a template-based advice response when model fails"""
        response = f"Legal Advice regarding: {query}\n\n"

        # Add introduction
        response += "Based on my analysis of relevant tax court cases, here is my legal advice:\n\n"

        # Identify issue
        response += "ISSUE:\n"
        response += "Your query involves tax filing obligations and potential consequences for missed deadlines.\n\n"

        # Add relevant law section
        response += "RELEVANT LAW:\n"
        for i, case in enumerate(relevant_cases[:2]):  # Use top 2 cases
            case_id = case['case_id']
            case = self.case_database.get_case(case_id)
            response += f"• In {case.name} ({case.year}), the court established that timely filing is essential, and penalties may apply for late submissions.\n"

        # Analysis
        response += "\nANALYSIS:\n"
        response += "Failure to file tax returns by the required deadline can result in penalties, interest charges, and potential legal consequences. The tax court has consistently held that taxpayers are responsible for timely compliance with filing requirements.\n\n"

        # Advice
        response += "ADVICE:\n"
        response += "1. File your returns immediately to minimize additional penalties and interest.\n"
        response += "2. Consider requesting an abatement of penalties if you have reasonable cause for the delay.\n"
        response += "3. If you owe taxes, pay as much as possible now to reduce interest charges.\n"
        response += "4. Consider consulting with a tax professional for assistance with your specific situation.\n"
        response += "5. Maintain records of all communications with tax authorities regarding this matter.\n\n"

        response += "Note: This advice is based on general tax court precedents and should not substitute for personalized legal counsel."

        return {
            "response": response,
            "relevant_cases": relevant_cases
        }

    def generate_template_arguments(self, query, relevant_cases):
        """Generate a template-based legal arguments response when model fails"""
        case_name = query if "v." in query else "the referenced tax matter"

        response = f"Legal Arguments for {case_name}\n\n"

        # Add key facts
        response += "KEY FACTS:\n"
        response += "This case involves a tax dispute between the parties, likely concerning property assessment, tax liability, or exemption status.\n\n"

        # Legal issues
        response += "LEGAL ISSUES:\n"
        response += "1. What are the proper criteria for determining tax liability in this situation?\n"
        response += "2. Does the taxpayer qualify for any exemptions or special considerations?\n"
        response += "3. Were proper procedures followed in the assessment process?\n\n"

        # Plaintiff arguments
        response += "PLAINTIFF'S ARGUMENTS:\n"
        for i, case in enumerate(relevant_cases[:2]):
            case_id = case['case_id']
            case = self.case_database.get_case(case_id)
            response += f"• Citing {case.name}, the plaintiff could argue that the assessment did not follow proper procedural requirements, potentially invalidating the tax determination.\n"
        response += "• The plaintiff might contend that the property qualifies for an exemption under relevant tax statutes.\n\n"

        # Defendant arguments
        response += "DEFENDANT'S ARGUMENTS:\n"
        for i, case in enumerate(relevant_cases[:1]):
            case_id = case['case_id']
            case = self.case_database.get_case(case_id)
            response += f"• Relying on {case.name}, the defendant could maintain that the assessment was conducted according to established guidelines and precedent.\n"
        response += "• The defendant might argue that the plaintiff does not meet the statutory requirements for the claimed exemptions or relief.\n\n"

        # Likely outcome
        response += "LIKELY OUTCOME:\n"
        response += "Based on the available precedents, the outcome will likely depend on the specific facts and circumstances of this case, particularly regarding procedural compliance and whether statutory requirements for exemptions were satisfied. More detailed factual information would be needed for a more precise prediction.\n\n"

        response += "Note: These arguments are based on general tax court precedents and would need to be tailored to the specific details of the case."

        return {
            "response": response,
            "relevant_cases": relevant_cases
        }

    def generate_template_general(self, query, relevant_cases):
        """Generate a template-based general response when model fails"""
        response = f"Information regarding: {query}\n\n"

        # Add introduction
        response += "Based on relevant tax court cases, here is the information you requested:\n\n"

        # Add content from relevant cases
        for i, case_data in enumerate(relevant_cases[:3]):
            case_id = case_data['case_id']
            case = self.case_database.get_case(case_id)
            response += f"According to {case.name} ({case.year}):\n"

            # Try to extract a relevant excerpt
            case_text = case.text.lower()
            query_terms = query.lower().split()

            relevant_excerpt = ""
            sentences = self.re.split(r'(?<=[.!?])\s+', case.text)

            for sentence in sentences[:20]:  # Check first 20 sentences
                if any(term in sentence.lower() for term in query_terms):
                    relevant_excerpt = sentence
                    break

            if relevant_excerpt:
                response += f"• {relevant_excerpt}\n\n"
            else:
                response += f"• This case addresses matters related to tax assessment, liability, and procedural requirements that may be relevant to your query.\n\n"

        # Add conclusion
        response += "SUMMARY:\n"
        response += "The tax court precedents suggest that matters involving tax filings, assessments, and liabilities are governed by specific procedural requirements and statutory provisions. Compliance with deadlines and proper documentation are typically essential factors in tax court decisions.\n\n"

        response += "For more specific guidance on your situation, consider consulting with a tax professional who can provide advice tailored to your circumstances."

        return {
            "response": response,
            "relevant_cases": relevant_cases
        }

    def generate_simple_response(self, query, mode, relevant_cases):
        """Generate a structured rule-based response when ML models fail.
        This function serves as a critical fallback mechanism to ensure the system
        always provides useful information without requiring ML resources."""

        # Extract more information from relevant cases
        structured_case_info = []
        legal_principles = set()  # Use a set to avoid duplicates

        for case_data in relevant_cases:
            case_id = case_data['case_id']
            case = self.case_database.get_case(case_id)
            case_text = case.text

            # Extract potential principles from case
            principles = self.extract_legal_principles(case_text)
            legal_principles.update(principles)

            # Find relevant sections/sentences
            relevant_content = {
                'name': case.name,
                'year': case.year,
                'facts': self.extract_section(case_text, ['FACTS', 'BACKGROUND', 'STATEMENT OF FACTS'], 500),
                'issue': self.extract_section(case_text, ['ISSUE', 'QUESTION PRESENTED'], 300),
                'holding': self.extract_section(case_text, ['HOLDING', 'CONCLUSION', 'JUDGMENT'], 500),
                'keywords': self.find_query_related_content(case_text, query, 2)  # Extract 2 relevant sentences
            }
            structured_case_info.append(relevant_content)

        # Create response based on mode
        if mode == "1":  # Advice
            return self.generate_legal_advice(query, structured_case_info, legal_principles)
        elif mode == "2":  # Arguments
            return self.generate_legal_arguments(query, structured_case_info, legal_principles)
        else:  # General Response
            return self.generate_general_information(query, structured_case_info, legal_principles)

    def extract_section(self, text, section_names, max_length=500):
        """Extract a specific section from case text based on common section headers"""
        for section_name in section_names:
            # Try to find the section and the text that follows until the next section
            pattern = rf'(?i){section_name}[:\s]+(.*?)(?=(?:[A-Z][A-Z\s]+:)|$)'
            match = self.re.search(pattern, text, self.re.DOTALL)
            if match:
                section_text = match.group(1).strip()
                # Truncate to max_length but try to end at a sentence
                if len(section_text) > max_length:
                    truncated = section_text[:max_length]
                    last_period = truncated.rfind('.')
                    if last_period > max_length * 0.7:  # If we found a period in the latter part
                        truncated = truncated[:last_period + 1]
                    return truncated
                return section_text
        return ""  # Return empty string if section not found

    def extract_legal_principles(self, text):
        """Extract potential legal principles from case text"""
        principles = set()

        # Look for sentences containing principle indicators
        indicators = [
            "principle", "rule", "held that", "court found", "established that",
            "doctrine", "standard", "test", "requirement", "factor", "concluded that",
            "determined that", "statutory", "regulation", "tax code", "section", "provision"
        ]

        sentences = self.re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in indicators):
                # Clean up the sentence
                clean_sentence = self.re.sub(r'\s+', ' ', sentence).strip()
                if len(clean_sentence) > 20:  # Avoid very short fragments
                    principles.add(clean_sentence)

        # Limit to 5 principles to keep it manageable
        return list(principles)[:5]

    def find_query_related_content(self, text, query, num_sentences=2):
        """Find sentences in the text that are most relevant to the query"""
        query_terms = query.lower().split()
        sentences = self.re.split(r'(?<=[.!?])\s+', text)

        # Score each sentence based on how many query terms it contains
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for term in query_terms if term.lower() in sentence.lower())
            if score > 0:
                scored_sentences.append((sentence, score))

        # Sort by score and take the top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_sentences[:num_sentences]]

    def generate_legal_advice(self, query, case_info, legal_principles):
        """Generate structured legal advice based on case information"""
        response = f"Legal Advice regarding '{query}':\n\n"

        # 1. ISSUE section
        response += "ISSUE:\n"
        if "tax" in query.lower():
            response += "Your query concerns tax obligations and potential legal implications. "
        elif "property" in query.lower():
            response += "Your query concerns property tax assessment and potential challenges. "
        elif "filing" in query.lower() or "return" in query.lower():
            response += "Your query concerns tax filing requirements and potential consequences for non-compliance. "
        else:
            response += "Your query concerns tax court matters that may have legal and financial implications. "

        # Add query-specific context
        if "not filed" in query.lower() or "late" in query.lower():
            response += "Specifically, you're asking about the consequences and remedies for late or unfiled tax returns.\n\n"
        elif "assessment" in query.lower():
            response += "Specifically, you're asking about tax assessment procedures and potential grounds for appeal.\n\n"
        else:
            response += "Understanding the applicable legal standards is crucial to resolving your situation.\n\n"

        # 2. RELEVANT LAW section
        response += "RELEVANT LAW:\n"
        if legal_principles:
            for i, principle in enumerate(legal_principles[:3]):  # Limit to 3 principles
                response += f"• {principle}\n"
        else:
            # Fallback if no principles were found
            response += "• Tax filings must generally be submitted by statutory deadlines unless extensions are granted.\n"
            response += "• Late filings may result in penalties and interest charges that accumulate over time.\n"
            response += "• The tax court may consider reasonable cause arguments for penalty abatement in certain circumstances.\n"
        response += "\n"

        # 3. ANALYSIS section
        response += "ANALYSIS:\n"
        if case_info:
            for i, case in enumerate(case_info[:2]):  # Use top 2 cases
                response += f"In {case['name']} ({case['year']}), "

                # Use case-specific content if available
                if case['keywords']:
                    response += f"the court addressed a similar situation and found that {case['keywords'][0]}\n"
                elif case['holding']:
                    response += f"the court ruled that {case['holding'][:150]}...\n"
                else:
                    response += "the court emphasized the importance of timely compliance with tax obligations.\n"
        else:
            response += "Based on tax court precedent, timely filing and proper documentation are critical elements in tax compliance matters. The burden typically falls on the taxpayer to demonstrate compliance or establish reasonable cause for failure to comply.\n"
        response += "\n"

        # 4. ADVICE section
        response += "ADVICE:\n"
        if "not filed" in query.lower() or "late" in query.lower():
            response += "1. File your tax returns immediately to minimize additional penalties and interest.\n"
            response += "2. If you have reasonable cause for the delay, prepare documentation to support a penalty abatement request.\n"
            response += "3. Consider making any tax payments as soon as possible, as penalties for late payment accumulate separately.\n"
            response += "4. Maintain records of all communications with tax authorities.\n"
        elif "assessment" in query.lower():
            response += "1. Review the assessment notice carefully to understand the basis for the determination.\n"
            response += "2. Gather documentation that supports your position if you believe the assessment is incorrect.\n"
            response += "3. File an appeal within the statutory timeframe (typically 45-90 days).\n"
            response += "4. Consider consulting with a tax professional for assistance with the appeal process.\n"
        else:
            response += "1. Ensure all required tax documents are filed in accordance with applicable deadlines.\n"
            response += "2. Maintain comprehensive records to substantiate your tax positions.\n"
            response += "3. Consider seeking professional tax assistance to evaluate your specific situation.\n"
            response += "4. Be proactive in addressing any notices or inquiries from tax authorities.\n"

        response += "\nThis guidance is based on general tax court principles and not on your specific circumstances. For personalized advice, please consult with a qualified tax attorney or tax professional."

        return {
            "response": response,
            "relevant_cases": [case['name'] for case in case_info]
        }

    def generate_legal_arguments(self, query, case_info, legal_principles):
        """Generate structured legal arguments for both sides based on case information"""
        # Determine if query contains a case name
        case_name = query if "v." in query else "the tax matter in question"

        response = f"Legal Arguments regarding '{case_name}':\n\n"

        # 1. KEY FACTS section
        response += "KEY FACTS:\n"
        facts_added = False

        for case in case_info:
            if case['facts']:
                facts = case['facts'][:200]  # Limit length
                response += f"Based on similar cases like {case['name']}, relevant facts might include: {facts}...\n"
                facts_added = True
                break

        if not facts_added:
            # Generate generic facts based on query terms
            if "property" in query.lower() or "assessment" in query.lower():
                response += "This case likely involves a dispute over property tax assessment methodology or valuation.\n"
            elif "exemption" in query.lower():
                response += "This case appears to concern eligibility for tax exemptions or relief provisions.\n"
            elif "filing" in query.lower() or "return" in query.lower():
                response += "This case concerns compliance with tax filing requirements and potential penalties.\n"
            else:
                response += "This case involves a tax dispute that requires analysis of relevant statutes and precedents.\n"
        response += "\n"

        # 2. LEGAL ISSUES section
        response += "LEGAL ISSUES:\n"
        issues_found = False

        for case in case_info:
            if case['issue']:
                response += f"Similar to {case['name']}, key issues may include: {case['issue']}\n"
                issues_found = True
                break

        if not issues_found:
            # Generate generic issues based on query
            if "property" in query.lower() or "assessment" in query.lower():
                response += "1. Was the property valuation conducted using proper methodologies and comparable properties?\n"
                response += "2. Did the assessment comply with statutory requirements and established procedures?\n"
            elif "exemption" in query.lower():
                response += "1. Does the entity or property qualify for the claimed tax exemption under applicable law?\n"
                response += "2. Were all procedural requirements for claiming the exemption properly followed?\n"
            else:
                response += "1. Was there compliance with applicable tax code provisions and regulations?\n"
                response += "2. If non-compliance occurred, are there valid grounds for relief or mitigation?\n"
        response += "\n"

        # 3. PLAINTIFF'S ARGUMENTS section
        response += "PLAINTIFF'S ARGUMENTS:\n"
        if legal_principles:
            response += f"1. Citing the principle established in recent cases that '{legal_principles[0][:150]}...', the plaintiff could argue that this standard applies favorably to their position.\n\n"

        if case_info:
            for i, case in enumerate(case_info[:2]):
                if case['holding'] or case['keywords']:
                    content = case['holding'] if case['holding'] else case['keywords'][0] if case['keywords'] else ""
                    response += f"2. In {case['name']} ({case['year']}), the court held that {content[:150]}... The plaintiff could leverage this precedent to argue that similar considerations apply here.\n\n"
                    break

        if not legal_principles and not case_info:
            response += "1. The plaintiff likely argues that the tax determination fails to properly apply relevant statutes and regulations.\n"
            response += "2. Procedural deficiencies in the assessment process may be cited as grounds for relief.\n"
        response += "\n"

        # 4. DEFENDANT'S ARGUMENTS section
        response += "DEFENDANT'S ARGUMENTS:\n"
        if legal_principles and len(legal_principles) > 1:
            response += f"1. The defendant could counter by emphasizing that '{legal_principles[1][:150]}...', which supports their position on the tax matter.\n\n"

        if case_info and len(case_info) > 1:
            case = case_info[1]  # Use a different case for the defendant
            content = case['holding'] if case['holding'] else case['keywords'][0] if case['keywords'] else ""
            if content:
                response += f"2. Referring to {case['name']} ({case['year']}), the defendant could argue that {content[:150]}... This precedent supports upholding the tax determination.\n\n"

        if not legal_principles and (not case_info or len(case_info) <= 1):
            response += "1. The defendant would likely maintain that the tax determination was made in accordance with established procedures and statutory requirements.\n"
            response += "2. The burden of proof typically rests with the taxpayer to demonstrate that the determination was incorrect.\n"
        response += "\n"

        # 5. POTENTIAL OUTCOME section
        response += "POTENTIAL OUTCOME:\n"
        response += "The outcome will likely depend on specific facts and evidence presented, particularly regarding:\n"
        response += "• Whether procedural requirements were properly followed by both parties\n"
        response += "• The weight of precedential cases most analogous to the current situation\n"
        response += "• The clarity of applicable statutory provisions and regulations\n\n"

        response += "Without more specific details about this case, a definitive prediction is not possible. The tax court typically focuses on both procedural compliance and substantive application of tax laws in rendering decisions."

        return {
            "response": response,
            "relevant_cases": [case['name'] for case in case_info]
        }

    def generate_general_information(self, query, case_info, legal_principles):
        """Generate general information response based on case information"""
        response = f"Information regarding '{query}':\n\n"

        # Add introduction based on query
        if "property" in query.lower() or "assessment" in query.lower():
            response += "Your query relates to property tax assessment procedures and legal standards in tax court cases.\n\n"
        elif "exemption" in query.lower():
            response += "Your query concerns tax exemptions and the legal standards applied by tax courts in determining eligibility.\n\n"
        elif "filing" in query.lower() or "return" in query.lower():
            response += "Your query relates to tax filing requirements and potential consequences for non-compliance.\n\n"
        else:
            response += "Your query relates to tax court procedures and precedents that may be relevant to your situation.\n\n"

        # Add information from relevant cases
        response += "KEY INFORMATION FROM RELEVANT CASES:\n"
        if case_info:
            for i, case in enumerate(case_info[:3]):  # Limit to top 3 cases
                response += f"{i + 1}. {case['name']} ({case['year']}): "

                if case['keywords']:
                    response += f"{case['keywords'][0]}\n"
                elif case['holding']:
                    response += f"{case['holding'][:200]}...\n"
                else:
                    response += "This case established important precedent for tax matters similar to your query.\n"
        else:
            response += "No specific cases directly matching your query were found in the database.\n"
        response += "\n"

        # Add relevant legal principles
        response += "RELEVANT LEGAL PRINCIPLES:\n"
        if legal_principles:
            for i, principle in enumerate(legal_principles[:3]):  # Limit to 3 principles
                response += f"• {principle[:200]}...\n"
        else:
            response += "• Tax courts generally require taxpayers to demonstrate compliance with statutory requirements.\n"
            response += "• Procedural compliance is often as important as substantive tax positions in court decisions.\n"
            response += "• The burden of proof in many tax matters rests with the taxpayer rather than the taxing authority.\n"
        response += "\n"

        # Add conclusion
        response += "SUMMARY:\n"
        response += "The tax court precedents indicate that matters involving tax assessments, filings, and exemptions are governed by specific statutory requirements and procedural rules. Understanding these requirements is essential for effectively addressing tax matters before the court.\n\n"

        response += "For more specific guidance on your particular situation, consider consulting the full text of relevant cases or seeking advice from a qualified tax professional."

        return {
            "response": response,
            "relevant_cases": [case['name'] for case in case_info]
        }


# Initialize the LexDebate AI system
case_db = CaseDatabase()
case_embeddings = CaseEmbeddings()
legal_rag = None


def init_system():
    global legal_rag

    # Load existing database if available
    cases = case_db.load_database()

    # Load embeddings if available
    embeddings = case_embeddings.load_embeddings()

    # Initialize RAG system if data is available
    if cases and embeddings:
        legal_rag = LegalRAG(case_db, case_embeddings)
        return True
    else:
        return False


# Initialize system on startup
system_ready = init_system()


# Function to build database in background
def build_database_task():
    global system_ready, legal_rag

    # Create database
    cases = case_db.create_database()

    # Create embeddings
    if cases:
        embeddings = case_embeddings.create_embeddings(cases)
        if embeddings:
            legal_rag = LegalRAG(case_db, case_embeddings)
            system_ready = True


# HTML Template
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LexDebate AI</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .button { 
                padding: 10px 15px; 
                background-color: #4CAF50; 
                color: white; 
                border: none; 
                cursor: pointer; 
                margin: 10px 5px;
            }
            .chat-container {
                height: 400px;
                border: 1px solid #ddd;
                padding: 10px;
                overflow-y: auto;
                margin-bottom: 10px;
            }
            textarea {
                width: 100%;
                height: 100px;
                margin-bottom: 10px;
            }
            #status {
                padding: 10px;
                background-color: #f8f9fa;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LexDebate AI</h1>
            <div id="status">Status: Checking...</div>

            <form id="database-form" action="/build_database" method="post">
                <button type="submit" class="button">Build/Update Database</button>
            </form>

            <div id="chat-area" style="display:none;">
                <div class="chat-container" id="chat-container">
                    <p><strong>LexDebate AI:</strong> Hello! Ask me a question about tax law.</p>
                </div>

                <textarea id="query-input" placeholder="Enter your query..."></textarea>

                <button id="advice-btn" class="button">Legal Advice</button>
                <button id="arguments-btn" class="button">Legal Arguments</button>
                <button id="general-btn" class="button">Other</button>
                <button id="send-btn" class="button">Send</button>
            </div>
        </div>

        <script>
            $(document).ready(function() {
                // Check status periodically
                checkStatus();
                setInterval(checkStatus, 5000);

                // Use a regular form submission instead of AJAX
                $("#database-form").submit(function(e) {
                    e.preventDefault();
                    $("#status").text("Building database...");

                    $.post("/build_database", function(data) {
                        $("#status").text("Database build initiated. Check status for updates.");
                    });
                });

                // Set mode and send query
                let currentMode = "1";

                $("#advice-btn").click(function() { currentMode = "1"; $(this).css('background-color', '#2E8B57'); });
                $("#arguments-btn").click(function() { currentMode = "2"; $(this).css('background-color', '#2E8B57'); });
                $("#general-btn").click(function() { currentMode = "3"; $(this).css('background-color', '#2E8B57'); });

                $("#send-btn").click(sendQuery);

                function sendQuery() {
                    const query = $("#query-input").val();
                    if (!query) return;

                    $("#chat-container").append("<p><strong>You:</strong> " + query + "</p>");
                    $("#chat-container").append("<p><strong>LexDebate AI:</strong> Thinking...</p>");
                    $("#query-input").val("");

                    $.ajax({
                        url: "/query",
                        type: "POST",
                        contentType: "application/json",
                        data: JSON.stringify({ query: query, mode: currentMode }),
                        success: function(data) {
                            // Remove the "thinking" message
                            $("#chat-container p:last").remove();

                            // Add the response
                            $("#chat-container").append("<p><strong>LexDebate AI:</strong> " + data.response + "</p>");

                            // Add case info
                            if (data.cases && data.cases.length > 0) {
                                let caseInfo = "<p><small>Based on: ";
                                data.cases.forEach(function(c) {
                                    caseInfo += c.name + " (" + c.year + "), ";
                                });
                                caseInfo = caseInfo.slice(0, -2) + "</small></p>";
                                $("#chat-container").append(caseInfo);
                            }

                            // Scroll to bottom
                            $("#chat-container").scrollTop($("#chat-container")[0].scrollHeight);
                        },
                        error: function() {
                            // Remove the "thinking" message
                            $("#chat-container p:last").remove();
                            $("#chat-container").append("<p><strong>LexDebate AI:</strong> Sorry, there was an error processing your request.</p>");
                        }
                    });
                }

                function checkStatus() {
                    $.get("/check_status", function(data) {
                        if (data.building) {
                            $("#status").text("Building database: " + data.message + " (" + data.progress + "%)");
                        } else if (data.system_ready) {
                            $("#status").text("System ready.");
                            $("#chat-area").show();
                        } else {
                            $("#status").text("System not ready. Please build the database.");
                            $("#chat-area").hide();
                        }
                    });
                }
            });
        </script>
    </body>
    </html>
    '''

# Routes
@app.route('/build_database', methods=['POST'])
def build_database():
    global database_status

    if not database_status["building"]:
        database_status["building"] = True
        database_status["message"] = "Starting database build..."
        database_status["progress"] = 0

        # Start database build in background
        thread = threading.Thread(target=build_database_task)
        thread.daemon = True
        thread.start()

    return jsonify({
        "status": "started",
        "message": "Database build started"
    })


@app.route('/check_status')
def check_status():
    global database_status, system_ready

    return jsonify({
        "building": database_status["building"],
        "message": database_status["message"],
        "progress": database_status["progress"],
        "system_ready": system_ready
    })


@app.route('/query', methods=['POST'])
def process_query():
    global system_ready, legal_rag

    if not system_ready:
        return jsonify({
            "error": "System not ready. Please build the database first."
        })

    data = request.json
    query = data.get('query', '')
    mode = data.get('mode', '3')  # Default to general mode

    if not query:
        return jsonify({
            "error": "Query cannot be empty"
        })

    # Find relevant cases
    relevant_cases = legal_rag.find_relevant_cases(query)

    # Generate response
    result = legal_rag.generate_response(query, mode, relevant_cases)

    # Format case information for display
    cases_info = []
    for i, case in enumerate(result["relevant_cases"]):
        cases_info.append({
            "name": case["name"],
            "year": case["year"],
            "similarity": round(float(case["similarity"]) * 100, 2)
        })

    return jsonify({
        "response": result["response"],
        "cases": cases_info
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)