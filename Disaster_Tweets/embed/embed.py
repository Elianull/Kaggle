import pandas as pd
from angle_emb import AnglE
import torch
import numpy as np
import sys

def format_row_as_string(row):
    row_string = f"Tweet ID: {row['id']}, "
    row_string += f"Keyword: {row['keyword'] if pd.notnull(row['keyword']) else 'N/A'}, "
    row_string += f"Location: {row['location'] if pd.notnull(row['location']) else 'N/A'}, "
    row_string += f"Text: {row['text']}."
    return row_string

def main(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Apply format_row_as_string to each row
    formatted_strings = df.apply(format_row_as_string, axis=1).tolist()

    # Initialize the AnglE model
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

    # Function to encode summaries using AnglE
    def encode(strings, batch_size=100):
        all_embeddings = []
        for i in range(0, len(strings), batch_size):
            batch = [string for string in strings[i:i + batch_size] if string]
            batch_embeddings = angle.encode(batch, to_numpy=True)
            all_embeddings.append(batch_embeddings)
        encoded_vectors = np.concatenate(all_embeddings, axis=0)
        return encoded_vectors

    # Pass the formatted strings to the encode function
    encoded_vectors = encode(formatted_strings)
    print("Encoding Complete.")
    np.save("data/encoded_vectors.npy", encoded_vectors)
    print("Encoded vectors saved to data/encoded_vectors.npy")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python embed.py <path_to_csv>")
    else:
        main(sys.argv[1])
