import os
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.graphs import Neo4jGraph
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
import pprint
from bs4 import BeautifulSoup

pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)       # Adjust the width to avoid wrapping



llm = ChatOpenAI(openai_api_key="",
             model='gpt-4o-mini',
             temperature=0.7,
             max_tokens=100 )

# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "welcome@123"

# Step 1: Set up Neo4j connection
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Step 2: Initialize models and tokenizers with vector index names
models = {
    # "roberta-base": {
    #     "name": "xlm-roberta-base",
    #     "index_name": "roberta_base_embedding",  # Updated index name
    #     "tokenizer": None,
    #     "model": None,
    # },
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "index_name": "mbert_embedding",  # Updated index name
        "tokenizer": None,
        "model": None,
    },
    # "mdeberta-v3": {
    #     "name": "microsoft/mdeberta-v3-base",
    #     "index_name": "mdeberta_v3_embedding",  # Updated index name
    #     "tokenizer": None,
    #     "model": None,
    # },
    "labse": {
        "name": "sentence-transformers/LaBSE",
        "index_name": "labse_embedding",  # Updated index name
        "tokenizer": None,
        "model": None,
    },
    # "distilmBERT": {
    #     "name": "distilbert-base-multilingual-cased",
    #     "index_name": "distilmBERT_embedding",  # Updated index name
    #     "tokenizer": None,
    #     "model": None,
    # },
    # "mt5": {
    #     "name": "google/mt5-base",
    #     "index_name": "mt5_embedding",  # Updated index name
    #     "tokenizer": None,
    #     "model": None,
    # },
    # "xlm": {
    #     "name": "xlm-mlm-tlm-xnli15-1024",
    #     "index_name": "xlm_embedding",  # Updated index name
    #     "tokenizer": None,
    #     "model": None,
    # }
}

# Load each model and tokenizer
for key, model_info in models.items():
    print(f"Loading model: {model_info['name']}")
    model_info["tokenizer"] = AutoTokenizer.from_pretrained(model_info["name"])
    model_info["model"] = AutoModel.from_pretrained(model_info["name"])

def summarize_text(input_text):
    prompt_template = PromptTemplate(template="""
    You are an intelligent assistant tasked with summarizing website content.
    Your role is to generate a concise and clear summary of the given text for contextual analysis.

    Text: {text}
    Summary:""", input_variables=["text"])

    response = llm.invoke(prompt_template.format(text=input_text[:128000]))
    #print("RESponse:",response.content)
    return response.content.strip()


def query_neo4j_graph(embedding, index_name, top_n=6):
    """
    Query the Neo4j graph for nodes with embeddings similar to the given embedding.
    """
    query = f"""
        CALL db.index.vector.queryNodes('{index_name}', $top_n, $embedding)
    YIELD node, score
    WHERE node.labse_weight_alpha1_5 > 0.5 OR node.mbert_weight_alpha1_5 > 0.5
    WITH node.name AS name_label, score, node.labse_weight_alpha1_5 as labse_weight, node.mbert_weight_alpha1_5 as mbert_weight
    CALL {{
        WITH name_label
        MATCH (n:Entity)
        WHERE n.name = name_label
        OPTIONAL MATCH (n)<-[:SIMILAR_TO]-(w:Word)
        OPTIONAL MATCH (n)<-[:CONTAINS]-(t3:Tier3)
        OPTIONAL MATCH (n)<-[:CONTAINS]-(t2:Tier2)
        OPTIONAL MATCH (n)<-[:CONTAINS]-(t1:Tier1)
        WITH n, w, t3, t2, t1
        RETURN 
            CASE 
                WHEN w IS NOT NULL THEN w
                WHEN t3 IS NOT NULL THEN t3
                WHEN t2 IS NOT NULL THEN t2
                WHEN t1 IS NOT NULL THEN t1
                ELSE n
            END AS first_node
    }}
    CALL {{
        // Subquery Part 2: Classify the first node
        WITH first_node
        OPTIONAL MATCH (first_node)<-[:CONTAINS]-(t3:Tier3)
        OPTIONAL MATCH (first_node)<-[:CONTAINS]-(t2:Tier2)
        OPTIONAL MATCH (first_node)<-[:CONTAINS]-(t1:Tier1)
        RETURN 
            CASE 
                WHEN first_node:Word AND t3 IS NOT NULL THEN t3
                WHEN first_node:Word AND t2 IS NOT NULL THEN t2
                WHEN first_node:Word AND t1 IS NOT NULL THEN t1
                ELSE first_node
            END AS classified_node
    }}
    CALL {{
        // Subquery Part 3: Construct hierarchy based on classified_node
        WITH classified_node
        OPTIONAL MATCH (classified_node)<-[:IS_PARENT]-(t2:Tier2)
        OPTIONAL MATCH (classified_node)<-[:IS_PARENT]-(t1:Tier1)
        OPTIONAL MATCH (t2)<-[:IS_PARENT]-(t21:Tier1)
        RETURN 
            CASE 
                WHEN classified_node:Tier3 THEN classified_node.name
                ELSE NULL
            END AS tier3,
            CASE 
                WHEN classified_node:Tier3 THEN t2.name
                WHEN classified_node:Tier2 THEN classified_node.name
                ELSE NULL
            END AS tier2,
            CASE 
                WHEN classified_node:Tier3 THEN t21.name
                WHEN classified_node:Tier2 THEN t1.name
                WHEN classified_node:Tier1 THEN classified_node.name
                ELSE NULL
            END AS tier1
    }}
    RETURN name_label AS text, score,(labse_weight+mbert_weight) as weight, tier3, tier2, tier1
    """
    result = graph.query(query, {"embedding": embedding, "top_n": top_n})
    return result

def generate_embedding(text, tokenizer, model, model_name):
    """
    Generate embedding for a given text using the specified tokenizer and model.
    Handles specific requirements for models like mt5.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Special handling for mt5
    if model_name == "google/mt5-base":
        # Use a simple start token as the decoder input
        decoder_input_ids = torch.tensor([[0]])  # `0` is usually the default start token ID
        with torch.no_grad():
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
    else:
        with torch.no_grad():
            outputs = model(**inputs)

    # Use mean pooling of the last hidden state for the embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding
#
# def generate_embedding(text, tokenizer, model, model_name):
#     """
#     Generate a mean-pooled embedding for the given text using the model and tokenizer.
#     This version uses the attention mask to ensure that padding tokens do not affect the embedding.
#     """
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     embeddings = outputs.last_hidden_state
#     attention_mask = inputs['attention_mask']
#
#     # Mean pooling over non-padding tokens:
#     mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
#     sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
#     sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
#     mean_pooled = sum_embeddings / sum_mask
#
#     return mean_pooled.squeeze().tolist()


def process_results(df):
    def process_tiers(tier_column):
        tier_df = df[df[tier_column].notnull()]
        grouped = tier_df.groupby(tier_column).agg(
            total_votes=("vote", "sum"),
            confidence_measure=("vote", lambda x: x.sum() / df["vote"].sum()),
            average_similarity=("score", "mean"),
            average_weightage = ("weight", "mean")
        ).reset_index()
        return grouped

    tier3_results = process_tiers("tier3")
    tier2_results = process_tiers("tier2")
    tier1_results = process_tiers("tier1")

    return tier3_results, tier2_results, tier1_results


def merge_tiers(original, summary, tier_column, weight_original=0, weight_summary=1):
    merged = pd.merge(original, summary, on=tier_column, how="outer", suffixes=("_original", "_summary")).fillna(0)
   # pprint.pprint(merged.head(30))
    merged["similarity_final"] = (
        weight_original * merged["average_similarity_original"] +
        weight_summary * merged["average_similarity_summary"]
    )
    merged["total_votes"] = ((weight_original*merged["total_votes_original"]) +
                             (weight_summary *merged["total_votes_summary"]))
    merged["weightage_final"] = ((weight_original * merged["average_weightage_original"]) +
                             (weight_summary * merged["average_weightage_summary"]))
    #print(merged)
    return merged.sort_values(by=["weightage_final"], ascending=False).head(1)[tier_column].values[0]

# Main function to process both original and summarized text
def retrieve_and_process_results(input_text, top_n=6):
    print("Generating embeddings for the original text...")
    combined_results_original = []

    # Process original text
    for model_name, model_info in models.items():
        embedding = generate_embedding(input_text, model_info["tokenizer"], model_info["model"], model_info["name"])
        results = query_neo4j_graph(embedding, model_info["index_name"], top_n)
        # print(model_name, results)
        for result in results:
            combined_results_original.append({
                "model": model_name,
                "tier3": result.get("tier3"),
                "tier2": result.get("tier2"),
                "tier1": result.get("tier1"),
                "score": result.get("score"),
                "weight": result.get("weight"),
                "vote": 1,
            })

    # Convert to DataFrame
    # df_original = pd.DataFrame(combined_results_original)
    if not combined_results_original:
        df_original = pd.DataFrame(columns=["model", "tier3", "tier2", "tier1", "score", "weight", "vote"])
    else:
        df_original = pd.DataFrame(combined_results_original)
    print('******************df original ***************')
    print(df_original)

    # Summarize text
    print("Generating summary using OpenAI...")
    summarized_text = summarize_text(input_text)

    print(f"Summarized Text: {summarized_text}")

    # Process summarized text
    print("Generating embeddings for the summarized text...")
    combined_results_summary = []

    for model_name, model_info in models.items():
        embedding = generate_embedding(summarized_text, model_info["tokenizer"], model_info["model"], model_info["name"])
        results = query_neo4j_graph(embedding, model_info["index_name"], top_n)
        # print(model_name, results)
        for result in results:
            combined_results_summary.append({
                "model": model_name,
                "tier3": result.get("tier3"),
                "tier2": result.get("tier2"),
                "tier1": result.get("tier1"),
                "score": result.get("score"),
                "weight": result.get("weight"),
                "vote": 1,
            })

    # df_summary = pd.DataFrame(combined_results_summary)
    if not combined_results_summary:
        df_summary = pd.DataFrame(columns=["model", "tier3", "tier2", "tier1", "score", "weight", "vote"])
    else:
        df_summary = pd.DataFrame(combined_results_summary)
    print('******************df summary ***************')
    print(df_summary)

    # Process results for original and summarized text
    tier3_original, tier2_original, tier1_original = process_results(df_original)
    tier3_summary, tier2_summary, tier1_summary = process_results(df_summary)

    # print('******************tier 3 ***************')
    # print(tier3_original)
    # print(tier3_summary)
    #
    # final_tier3 = merge_tiers(tier3_original, tier3_summary, tier_column="tier3")
    # final_tier2 = merge_tiers(tier2_original, tier2_summary, tier_column="tier2")
    # final_tier1 = merge_tiers(tier1_original, tier1_summary, tier_column="tier1")

    # Process results for original and summarized text
    # Check if df_original or df_summary are empty before calling process_results
    if df_original.empty:
        # Provide a dummy result
        tier3_original, tier2_original, tier1_original = [pd.DataFrame(
            columns=["total_votes", "confidence_measure", "average_similarity", "average_weightage"])] * 3
    else:
        tier3_original, tier2_original, tier1_original = process_results(df_original)

    if df_summary.empty:
        # Provide a dummy result
        tier3_summary, tier2_summary, tier1_summary = [pd.DataFrame(
            columns=["total_votes", "confidence_measure", "average_similarity", "average_weightage"])] * 3
    else:
        tier3_summary, tier2_summary, tier1_summary = process_results(df_summary)

    # Now, before merging tiers, check if the DataFrames have data. If empty, return a default tier.
    def safe_merge_tiers(o, s, column):
        if o.empty and s.empty:
            return None  # or a default classification
        if o.empty:
            return s.sort_values(by=["total_votes", "average_weightage"], ascending=False).head(1)[column].values[0]
        if s.empty:
            return o.sort_values(by=["total_votes", "average_weightage"], ascending=False).head(1)[column].values[0]
        return merge_tiers(o, s, tier_column=column)

    final_tier3 = safe_merge_tiers(tier3_original, tier3_summary, "tier3")
    final_tier2 = safe_merge_tiers(tier2_original, tier2_summary, "tier2")
    final_tier1 = safe_merge_tiers(tier1_original, tier1_summary, "tier1")

    return final_tier3, final_tier2, final_tier1


def extract_text_from_html(file_path):
    """
    Extracts text content from an HTML file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def process_folder_structure(base_path):
    """
    Traverses the folder structure and processes HTML files based on the described hierarchy.
    """
    data = []
    count = 0
    print('inside folder structure')

    # Iterate through Tier 1 folders
    for tier1_folder in os.listdir(base_path):
        tier1_path = os.path.join(base_path, tier1_folder)
        if not os.path.isdir(tier1_path):
            continue

        tier1_name = tier1_folder.strip().lower().replace(":", "/")

        # Iterate through Tier 2 folders
        for tier2_folder in os.listdir(tier1_path):
            tier2_path = os.path.join(tier1_path, tier2_folder)
            if not os.path.isdir(tier2_path):
                continue

            tier2_name = tier2_folder.strip().lower().replace(":", "/")
            tier2_has_files = False  # Tracks if Tier 2 has any HTML files directly

            # Check contents of Tier 2
            for item in os.listdir(tier2_path):
                item_path = os.path.join(tier2_path, item)

                # If item is a Tier 3 folder
                if os.path.isdir(item_path):
                    tier3_name = item.strip().lower().replace(":", "/")
                    count += 1

                    # Process HTML files inside Tier 3
                    for html_file in os.listdir(item_path):
                        count +=1
                        if html_file.endswith(".html"):
                            html_file_path = os.path.join(item_path, html_file)
                            text_content = extract_text_from_html(html_file_path)
                            output_tier3, output_tier2, output_tier1 = retrieve_and_process_results(text_content, top_n=10)
                            # data.append({
                            #     "file_name": html_file,
                            #     "folder_tier1": tier1_name,
                            #     "folder_tier2": tier2_name,
                            #     "folder_tier3": tier3_name
                            # })
                            data.append({
                                "file_name": html_file,
                                "folder_tier1": tier1_name.strip().lower().replace(":", "/"),
                                "folder_tier2": tier2_name.strip().lower().replace(":", "/"),
                                "folder_tier3": tier3_name.strip().lower().replace(":", "/"),
                                "output_tier1": output_tier1,
                                "output_tier2": output_tier2,
                                "output_tier3": output_tier3,
                            })
                            # output_tier2.to_csv(str(count)+tier2_folder+tier1_folder+"original.csv", index=False)
                            # output_tier1.to_csv(str(count)+tier2_folder + tier1_folder + "summary.csv", index=False)

                # If item is an HTML file directly in Tier 2
                elif item.endswith(".html"):
                    count+=1
                    tier2_has_files = True
                    text_content = extract_text_from_html(item_path)
                    output_tier3, output_tier2, output_tier1 = retrieve_and_process_results(text_content, top_n=10)
                    # data.append({
                    #     "file_name": html_file,
                    #     "folder_tier1": tier1_name,
                    #     "folder_tier2": tier2_name,
                    #     "folder_tier3": tier3_name
                    # })
                    data.append({
                        "file_name": item,
                        "folder_tier1": tier1_name.strip().lower().replace(":", "/"),
                        "folder_tier2": tier2_name.strip().lower().replace(":", "/"),
                        "folder_tier3": None,
                        "output_tier1": output_tier1,
                        "output_tier2": output_tier2,
                        "output_tier3": output_tier3,
                    })
                    # output_tier2.to_csv(str(count)+tier2_folder + tier1_folder + "original.csv", index=False)
                    # output_tier1.to_csv(str(count)+tier2_folder + tier1_folder + "summary.csv", index=False)

            # Handle case where Tier 2 has no HTML files or Tier 3 folders
            if not tier2_has_files and not any(
                    os.path.isdir(os.path.join(tier2_path, sub)) for sub in os.listdir(tier2_path)):
                output_tier3, output_tier2, output_tier1 = retrieve_and_process_results(text_content, top_n=10)
                # data.append({
                #     "file_name": html_file,
                #     "folder_tier1": tier1_name,
                #     "folder_tier2": tier2_name,
                #     "folder_tier3": tier3_name
                # })
                data.append({
                    "file_name": None,
                    "folder_tier1": tier1_name.strip().lower().replace(":", "/"),
                    "folder_tier2": tier2_name.strip().lower().replace(":", "/"),
                    "folder_tier3": None,
                    "output_tier1": output_tier1,
                    "output_tier2": output_tier2,
                    "output_tier3": output_tier3,
                })
                # print(html_file)
                # print("folder_tier1", tier1_name.strip().lower().replace(":", "/"))
                # print("folder_tier2", tier2_name.strip().lower().replace(":", "/"))
                # print("folder_tier3", None)
                # output_tier2.to_csv(str(count)+tier2_folder + tier1_folder + "original.csv", index=False)
                # output_tier1.to_csv(str(count)+tier2_folder + tier1_folder + "summary.csv", index=False)

    return pd.DataFrame(data)


def save_to_csv(dataframe, output_file):
    """
    Saves the DataFrame to a CSV file.
    """
    try:
        dataframe.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")


if __name__ == "__main__":
    base_path = "/Users/divyaprasanthparaman/Documents/github/IAB_taxonomy_classification/neo4j_project/data/input/web_pages"  # Replace with your folder path
    output_csv = "output_tiers.csv"  # Output CSV file path

    print("Processing folder structure...")
    processed_data = process_folder_structure(base_path)

    print("Saving to CSV...")
    save_to_csv(processed_data, output_csv)

