import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


def create_database(output_path="C:/Users/nckee/PycharmProjects/WebScrapping/case_database.json"):
    # importing all the files to get the lists
    # then creating dictionaries out of them and storing it in JSON files

    from pdf_extractor import PDFExtractor
    from new import WebScrapping

    # Base setup
    BASE_URL = "https://law.justia.com/cases/new-jersey/tax-court/"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/117.0.0.0 Safari/537.36"
    }

    print("Starting to retrieve information from web scrapped data")
    scrapper = WebScrapping(headers=HEADERS, base_url=BASE_URL)
    extractor = PDFExtractor(headers=HEADERS, base_url=BASE_URL)

    # Step 2: Get year-wise links
    year_links = scrapper.get_year_links()
    print("Accessed year links")

    # Step 3: Get all case HTML links from year links
    html_links = scrapper.get_html_links(year_links)
    print("Accessed html links")

    # Step 4: Extract all PDF links from those HTML files
    pdf_links = extractor.get_pdf_links(html_links)
    print("Accessed pdf links")

    # Step 5: Extract content from each PDF
    all_text_cases = extractor.get_content(pdf_links)
    print("Accessed all_text_cases list")

    # Step 6: Extract case names
    case_names = extractor.get_name(all_text_cases)
    print("Accessed case names list")

    # Step 7: Extract decision years
    case_years = extractor.find_year(all_text_cases)
    print("Accessed case years list")

    # After accessing all the information create a dictionary
    cases = {}  # dict

    for i, (name, year, content) in tqdm(enumerate(zip(case_names, case_years, all_text_cases)),
                                         total=len(all_text_cases),
                                         desc="Processing cases"):
        # creating a case ID such as case_0, case_1
        case_id = f"case_{i}"
        cases[case_id] = {
            "name": name,
            "year": year,
            "Text": content[:50000]
        }

    # Save to JSON file
    print(f"Saving database with {len(cases)} cases to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Successfully created case database with {len(cases)} cases!")
    return cases

def embeddings_func(cases):

    e_model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = {} # case embeddings

    for key, value in cases.items(): # items returns a tuple value in the format (key, value)
        # most models have limit on tokens they can embed
        text = value['Text']
        # cases['Text'] -> wil try to access key from dictionary
        # embedding = e_model.encode(cases['Text'])
        text = text[:5000]
        embedding = e_model.encode(text)
        embeddings[key] = embedding

        # Save embeddings to a file to avoid recomputing them in the future
        print("Saving embeddings...")
        with open("C:/Users/nckee/PycharmProjects/WebScrapping/embeddings.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_embeddings = {k: v.tolist() for k, v in embeddings.items()}
            json.dump(serializable_embeddings, f)

        print(f"Successfully created embeddings for {len(embeddings)} cases!")
        return embeddings

    return embeddings
# redo  the below code.
def rag(query, mode, cases, embeddings):
    similarities = {} # a dict

    e_model = SentenceTransformer('all-MiniLM-L6-v2')
    for case_id, case_embeddings in embeddings.items():
        # embed the query
        similarity_embedding = e_model.encode(query)
        similarities[case_id] = similarity_embedding

        # finding similarity


if __name__ == "__main__":
    cases  = create_database()
    embeddings = embeddings_func(cases)

    input_text = input("Please enter the query you want to give to the model")
    mode_input = input("Please enter 1 if you want an advice, 2 if you want to generate legal arguments or 3 for other mode of response")

    rag(input_text, mode_input, cases, embeddings)
