import os
import sys

import pandas as pd

# Load the file into a pandas DataFrame
df = pd.read_csv(sys.argv[1])
# Build a dictionary with the "id" column as the key and "fav_count" as the value
id_to_score = dict(zip(df["id"], df["fav_count"]))
# Loop over all the directories specified on the command line
rows = {"POSTID": [], "IMAGEPATH": [], "SCORE": []}
for directory in sys.argv[2:]:
    if directory == "" or directory.endswith(os.path.sep):
        directory = os.path.dirname(directory)
    print(directory)
    # Loop over all the files in the directory
    for filename in os.listdir(directory):
        # Get the post_id, which is the filename without the extension
        post_id, ext = os.path.splitext(filename)
        if ext not in (".png", ".jpg", ".gif", ".webp"):
            continue
        post_id = int(post_id)
        # Look up the score in the dictionary
        if post_id in id_to_score:
            score = id_to_score[post_id]
        else:
            continue
        # Add the path and score to the results DataFrame
        path = os.path.join(directory, filename)
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        rows["POSTID"].append(post_id)
        rows["IMAGEPATH"].append(path)
        rows["SCORE"].append(score)
# Create a new DataFrame from the results
results = pd.DataFrame(data=rows)
# Save the results to a CSV file in the current directory
csv_filename = "dataset.csv"
results.to_csv(csv_filename, index=False)
