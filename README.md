IDEA:

The idea is to have an AI platform / website where legal information if available for users. LexDebate AI will be able to generate legal arguments as defense or prosecutor. The website is also intended to be useful
for citizens with less legal knowledge but need legal advice. LexDebate AI focused on tax cases. 

PROGRESS:

Until now, legal data has been scrapped from an official website - https://law.justia.com/cases/new-jersey/tax-court/

Part1

Web scrapping: Web scrapping has been performed with the help of BeautifulSoup library that allows to send get request to the website which enables us to access data. This website contains case files segregated to years from 1980 to 2025 spanning 44 years. 
The code written for web scrapping is successfully reading into all 44 years, which contain 1299 html files. The code is desgined to read the pdf files contained in the html files as well. 
Web scrapping is executed in two classes,
1. 'WebScrapping' which is responsible for accessing the year links from the base link and html links. 

2. 'pdf_extractor' is responsible for accessing pdf links from the html links. This class also contains functions that extracts all the texts from a pdf. It extracts case name - defendant and plantiff name, case year. Each legal document contains a small summary mentioning the legal proceedings in the entire case in the last few lines. This has been extracted as summary ( last 1500 characters)
   

Part2

Prepping_JSON: 

This file contains the following classes

LegalCase: Stores individual case data (ID, name, year, text).

CaseDatabase: Manages the collection of cases:

Creates database by scraping the tax court website.
Saves and loads cases from JSON files.
Provides case lookup functionality.


CaseEmbeddings: Handles vector representations of cases:

Creates embeddings using SentenceTransformer.
Saves and loads embeddings from JSON.
Provides query encoding for similarity search.


LegalRAG: 

Finds relevant cases based on query similarity -> implements the RAG function.
Generates structured responses using Flan-T5. But when the model fails to provide a lengthy enough response (=>50) or repeat the query then we have functions called fallback mechanisms to retrieve related content based on the query and answere based on that for reliable outputs. 
Formats responses for different modes (advice, arguments, general)

The system works by first collecting case data, converting it to embeddings for efficient retrieval, and then using Flan T5(LLM) to generate contextually relevant responses based on the most similar cases.

Part3, 
GUI

I built a website, http://127.0.0.1:5000
The website has 5 functional buttons, Legal advice button, legal argument generator, other button ( for general questions ), send button and Build database button. 

Instructions for a new user to use the files and website:

The Flask_website page imports the other modules and functions from those modules by creating an instance/objectt for the class. This file also contains the html css and JS script. 

The following are the versions you must have to be able to run the Flask_webpage file:

Python - 3.11

Flask - 2.3.3

Torch - 2.0.1

numpy - 1.24.3

sentence transformers - 2.2.2

Transformers - 4.30.2

tqdm - 4.65.0

scikit-learn - 1.2.2

requests - 2.31.0

BeautifulSoup4 - 4.12.2

PyMuPDF - 1.22.3

Note: Sometimes even when PyMuPDF is correctly installed you may encounter module 'frontend' not found. Please upgrade PyMUPDF if required. Otherwise, try 'from PyMuPDF import fitz' instead of 'import fitz'

A folder named data will be downloaded in the same directory as you are working on, where you can find cases retrieved in dictionary format ( a format that is required by the model )
Note: 
Processing 44 years, 1299 html files and every pdf file in all the 1299 html files takes a long time to load all the data. For demonstration purpose, I have included 2 recent years that is 2025 and 2024. It loads the case file it was able to access from these two year links. 
It embeds 3 cases (to enable faster execution to view and use the website).

