# Run this file from CMD/Terminal
# Example Command: python3 compute-sari.py original_file_name.txt pred_file_name.txt ref_file_name.txt


import sys
import evaluate

from evaluate import load
sari = load("sari")

target_original = sys.argv[1]  # Original file argument
target_pred = sys.argv[2]  # Pred file argument
target_ref = sys.argv[2]  # Ref file argument

# Open the test dataset human translation file and detokenize the references
originals = []

with open(target_original) as original:
    for line in original: 
        line = line.strip()
        originals.append(line)

print("Reference 1st sentence:", originals[0])

# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open(target_pred) as pred:  
    for line in pred: 
        line = line.strip()
        preds.append(line)

print("MTed 1st sentence:", preds[0])

# Open the test dataset human translation file and detokenize the references
refs = []

with open(target_ref) as ref:
    for line in ref: 
        line = line.strip()
        refs.append(line)

print("Reference 1st sentence:", refs[0])

refs = [refs]  # Yes, it is a list of list(s) as required by sacreBLEU

# Calculate and print the SARI score
sari_score = sari.compute(sources=sources, predictions=predictions, references=references)
print("SARI: ", sari_score)