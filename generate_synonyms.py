import time
import pandas as pd
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate

# OpenAI Setup
llm = ChatOpenAI(
    openai_api_key="",  # Replace with your OpenAI API key
    model="gpt-4",
    temperature=0.7,
    max_tokens=200
)


# Function to generate similar items
def generate_similar_items(keyword, tier1, tier2, tier3):
    prompt_template = PromptTemplate(template="""
    You are an expert in keyword research for SEO and content writing.
    Your task is to generate a list of related terms or synonyms that can be used for the given keyword in the context of media, websites, and articles.
    The related terms should be contextually relevant and optimized for the given tiers.

    Keyword: {keyword}
    Tier 1: {tier1}
    Tier 2: {tier2}
    Tier 3: {tier3}

    Related Terms:""", input_variables=["keyword", "tier1", "tier2", "tier3"])

    prompt = prompt_template.format(
        keyword=keyword,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3 if pd.notna(tier3) else "None"
    )

    response = llm.invoke(prompt)
    return response.content.strip()


# Main script to process the CSV
def process_csv(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Ensure the necessary columns exist
    if not {"keyword", "tier1", "tier2", "tier3"}.issubset(df.columns):
        raise ValueError("The input CSV file must contain 'keyword', 'tier1', 'tier2', and 'tier3' columns.")

    # Create a new column to store the related terms
    df["related_terms"] = None

    # Process each row and generate related terms
    for idx, row in df.iterrows():
        keyword = row["keyword"]
        tier1 = row["tier1"]
        tier2 = row["tier2"]
        tier3 = row["tier3"]

        print(f"Generating related terms for: {keyword} (Tier1: {tier1}, Tier2: {tier2}, Tier3: {tier3})")

        try:
            related_terms = generate_similar_items(keyword, tier1, tier2, tier3)
            df.at[idx, "related_terms"] = related_terms
        except Exception as e:
            print(f"Error generating related terms for {keyword}: {e}")
            df.at[idx, "related_terms"] = "Error"

        # Introduce a 2-second delay for every 10 rows
        if (idx + 1) % 10 == 0:
            print("Pausing for 2 seconds...")
            time.sleep(2)

    # Save the updated DataFrame to the output CSV file
    df.to_csv(output_file, index=False)
    print(f"Related terms saved to {output_file}")


# Run the script
if __name__ == "__main__":
    input_csv = "data_for_getting_synonyms_llm.csv"  # Replace with your input CSV file name
    output_csv = "output_with_related_terms_llm_generated.csv"  # Replace with your desired output CSV file name
    process_csv(input_csv, output_csv)
