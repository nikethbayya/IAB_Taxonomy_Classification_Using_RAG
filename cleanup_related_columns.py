import pandas as pd
import re

def clean_related_terms(input_file, output_file):
    # Read the input CSV
    df = pd.read_csv(input_file)

    # Ensure the `related_terms` column exists
    if "related_terms" not in df.columns:
        raise ValueError("The input CSV file must contain a 'related_terms' column.")

    # Clean up the `related_terms` column
    def clean_terms(terms):
        if pd.isna(terms):
            return ""
        # Replace newlines and other delimiters with a comma
        cleaned = terms.replace("\n", ", ").replace("\r", ", ").replace(".", ", ").replace("Tier 1:", "").replace("Tier 2:", "").replace("Tier 3:", "")
        # Remove numbers followed by commas (e.g., "1,", "2,")
        cleaned = re.sub(r"\d+\s*,\s*", "", cleaned)
        # Remove duplicate commas and strip any extra whitespace
        cleaned = ", ".join([term.strip() for term in cleaned.split(",") if term.strip()])
        return cleaned

    df["related_terms"] = df["related_terms"].apply(clean_terms)

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Cleaned related terms saved to {output_file}")

# Run the script
if __name__ == "__main__":
    input_csv = "output_with_related_terms_llm_generated.csv"  # Replace with your input CSV file name
    output_csv = "output_with_related_terms_llm_generated_cleaned.csv"  # Replace with your desired output CSV file name
    clean_related_terms(input_csv, output_csv)
