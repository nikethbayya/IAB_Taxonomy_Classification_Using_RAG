import pandas as pd

def flatten_words_and_synonyms(input_file, output_file):
    # Read the input CSV
    df = pd.read_csv(input_file)

    # Ensure the necessary columns exist
    if "word" not in df.columns or "synonyms" not in df.columns:
        raise ValueError("The input CSV file must contain 'word' and 'synonyms' columns.")

    # Extract the `word` column
    words = df["word"].dropna().tolist()

    # Extract and split the `synonyms` column into a list of words
    synonyms = df["synonyms"].dropna().str.split(",").tolist()

    # Flatten the synonyms list
    synonyms_flattened = [syn.strip() for sublist in synonyms for syn in sublist]

    # Combine words and synonyms into a single list
    all_unique_words = set(words + synonyms_flattened)

    # Create a new DataFrame
    result_df = pd.DataFrame({"unique_word": list(all_unique_words)})

    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Unique words and synonyms saved to {output_file}")

# Run the script
if __name__ == "__main__":
    input_csv = "output_with_related_terms_llm_generated_cleaned.csv"  # Replace with your input CSV file name
    output_csv = "output_with_related_terms_llm_generated_cleaned_flattened_keywords_only.csv"  # Replace with your desired output CSV file name
    flatten_words_and_synonyms(input_csv, output_csv)
