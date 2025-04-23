import json

from scipy.stats import cosine
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
    similarities = {} # a dict that holds case id as the key and a number that says how similar is that case with the query as the value

    e_model = SentenceTransformer('all-MiniLM-L6-v2')
    # embed the query
    query_embedding = e_model.encode(query)

    # after embedding the user query, find relevant cases from case_embeddings / embeddings -> dict
    # to find relevant cases from embeddings dictionary related to query
    for case_id, case_embeddings in embeddings.items():
        # finding similarity
        similarity =cosine(case_embeddings, query_embedding) # similarity is a number bw -1 to 1 that says how similar the case at the current iteration is with the query. if its near to 1 it has more similarity
        similarities[case_id] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse = True)

    # top 3 of the sorted similarities
    top_cases  = sorted_similarities[:3]
    # top_cases is a list of tuples,
    # [(case_id1, similarity_score1), (case_id2, similarity_score2), (case_id3, similarity_score3)]

    relevant_cases = [] # this is to hold the content ( non embedded form ) of cases relevant to the query
    for case_id, similarity in top_cases:
        # Now case_id contains the ID (e.g., "case_0")
        # and similarity contains the similarity score (e.g., 0.85)
        case = cases[case_id]
        relevant_cases.append({
            'name': case['name'],
            'year': case['year'],
            'similarity': similarity
        })
    return relevant_cases

# USING FLAN T5
def generate_response(query, mode, relevant_cases, cases):

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    # Initialize the Flan-T5 model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to("cpu")

    # Create context from relevant cases
    context = "Relevant cases:\n"
    for i, case in enumerate(relevant_cases):
        case_id = next(cid for cid, _ in relevant_cases if cases[cid]['name'] == case['name'])
        case_text = cases[case_id]['Text'][:3000]  # Get the first part of the case text
        context += f"{i + 1}. {case['name']} ({case['year']}): {case_text[:300]}...\n\n"

    # Create prompt based on mode
    if mode == "1":  # Advice
        instruction = f"Based on the following legal cases, provide legal advice for this query: {query}"
    elif mode == "2":  # Arguments
        instruction = f"Based on the following legal cases, generate legal arguments for both sides of this issue: {query}"
    else:  # Other
        instruction = f"Based on the following legal cases, respond to this query: {query}"

    input_text = f"{instruction}\n\n{context}"

    # Generate response
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024).to("cpu")
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "response": response,
        "relevant_cases": relevant_cases
    }


if __name__ == "__main__":
    cases  = create_database()
    embeddings = embeddings_func(cases)
    while True:

        input_text = input("Please enter the query you want to give to the model")
        mode_input = input("Please enter 1 if you want an advice, 2 if you want to generate legal arguments or 3 for other mode of response")

        relevant_cases = rag(input_text, mode_input, cases, embeddings)

        # Generate response
        result = generate_response(input_text, mode_input, relevant_cases, cases)

        # Display the response
        print("\nLexDebate AI Response:", result["response"])
        print("\nBased on relevant cases:")
        for i, case in enumerate(result["relevant_cases"]):
            print(f"  {i + 1}. {case['name']} ({case['year']}) - Relevance: {case['similarity']:.2f}")
