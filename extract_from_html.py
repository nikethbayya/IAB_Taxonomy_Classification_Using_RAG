import os
import pandas as pd
from bs4 import BeautifulSoup


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

                    # Process HTML files inside Tier 3
                    for html_file in os.listdir(item_path):
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

                # If item is an HTML file directly in Tier 2
                elif item.endswith(".html"):
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
                        "file_name": html_file,
                        "folder_tier1": tier1_name.strip().lower().replace(":", "/"),
                        "folder_tier2": tier2_name.strip().lower().replace(":", "/"),
                        "folder_tier3": tier3_name.strip().lower().replace(":", "/"),
                        "output_tier1": output_tier1,
                        "output_tier2": output_tier2,
                        "output_tier3": output_tier3,
                    })

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
                    "file_name": html_file,
                    "folder_tier1": tier1_name.strip().lower().replace(":", "/"),
                    "folder_tier2": tier2_name.strip().lower().replace(":", "/"),
                    "folder_tier3": tier3_name.strip().lower().replace(":", "/"),
                    "output_tier1": output_tier1,
                    "output_tier2": output_tier2,
                    "output_tier3": output_tier3,
                })

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
