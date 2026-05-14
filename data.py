from datasets import load_dataset
import pandas as pd

# Load the dataset from Hugging Face
ds = load_dataset("Rami/FAQ_student_accesiblity_for_UTD")

# Convert the "train" split to a pandas DataFrame and select specific columns
df = ds["train"].to_pandas()[["Question", "Answering"]]

# Save the extracted data to a CSV file
output_file = "faq_data.csv"
df.to_csv(output_file, index=False)
print(f"Data successfully pulled and saved to {output_file}")
