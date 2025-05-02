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


Part3, 
GUI

I built a website, http://127.0.0.1:5000
It has a drop down list that shows all the years accessed from the legal website. Upon clicking these years from the drop down, case details such as case name, case year and case summary will be displayed in the website. 
When the website is accessed, first page displayed is a loading page, it shows the status of data loading. The page automatically loads all the data once legal data has been scrapped. This website acts as a interface for a user to access the data scrapped off from the website. 

Note: 
Processing 44 years, 1299 html files and every pdf file in all the 1299 html files takes a long time to load all the data. For demonstration purpose, I have included 2 recent years that is 2025 and 2024. It loads the case file it was able to access from these two year links. 


Instructions for a new user to use the files and website

This github repository has four files. Save new.py, pdf_extractor.py, main.py and web_app.py in the same directory. Run the web_app.py to load data into the website and see how it runs. It has a clear debug information giving details after loading every case file. 
main.py can be run in order to see all the information in TUI. 
