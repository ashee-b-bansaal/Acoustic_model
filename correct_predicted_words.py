'''
Takes in word prediction from the model and uses language modeling for fixing word errors 
Example: tox -> top 
1/20/2024, Devansh Agarwal, da398@cornell.edu
'''

import argparse
import os 
import textdistance
# import Levenshtein
from ordered_set import OrderedSet

CORRECTED_FILE_NAME = "corrected_best_pred"
TXT = ".txt"

ITOS = {0: 'a',
 1: 'b',
 2: 'c',
 3: 'd',
 4: 'e',
 5: 'f',
 6: 'g',
 7: 'h',
 8: 'i',
 9: 'j',
 10: 'k',
 11: 'l',
 12: 'm',
 13: 'n',
 14: 'o',
 15: 'p',
 16: 'q',
 17: 'r',
 18: 's',
 19: 't',
 20: 'u',
 21: 'v',
 22: 'w',
 23: 'x',
 24: 'y',
 25: 'z'}

STOI = {'a': 0,
 'b': 1,
 'c': 2,
 'd': 3,
 'e': 4,
 'f': 5,
 'g': 6,
 'h': 7,
 'i': 8,
 'j': 9,
 'k': 10,
 'l': 11,
 'm': 12,
 'n': 13,
 'o': 14,
 'p': 15,
 'q': 16,
 'r': 17,
 's': 18,
 't': 19,
 'u': 20,
 'v': 21,
 'w': 22,
 'x': 23,
 'y': 24,
 'z': 25}



def get_vocabulary_set(path): 
    with open(path, 'r', encoding='utf-8') as f:
        vocab = f.read()
    return OrderedSet([w.lower() for w in vocab.split("\n")[:-1]])

def correct_predictions(pred_file_path, vocabulary, top_n):
    words_predicted_correctly_list = []
    words_predicted_incorrectly_list = []
    words_predicted_incorrectly_list.append("prediction, Ground Truth")
    with open(pred_file_path, 'r', encoding='utf-8') as f:
        preds = f.read()
    preds = preds.split("\n")[:-1]
    num_correct_words = 0
    corrected_words = []
    for i, pred in enumerate(preds):
        correct_pred_flag = False
        pred = pred.split(",")
        gt_word = pred[-2].lower()
        predicted_word = ""
        for ix in pred[-1].split():
            predicted_word += ITOS[int(ix)]
        pred.append(predicted_word + "(model predciction), correction:")
        distance_list = []
        if predicted_word in vocabulary:
            if predicted_word == gt_word:
                num_correct_words +=1
                words_predicted_correctly_list.append(gt_word)
                correct_pred_flag = True
            pred.append(predicted_word)
        else:

            for vocab_word in vocabulary:
                lev_distance = textdistance.algorithms.levenshtein.distance(predicted_word, vocab_word)
                distance_list.append((lev_distance, vocab_word))
            distance_list.sort(key=lambda x:x[0])
            top_n_words = [word for _, word in distance_list[:top_n]]
            if gt_word in top_n_words:
                num_correct_words +=1
                words_predicted_correctly_list.append(gt_word)
                correct_pred_flag = True
            pred.extend(top_n_words)
            for lev_word in top_n_words:
                pred.append(" ".join([str(STOI[ch]) for ch in lev_word]))
        if not correct_pred_flag:
            words_predicted_incorrectly_list.append((predicted_word, gt_word))
        corrected_words.append(",".join(pred))
    print("Number of Correct Words = ", num_correct_words)
    print("Correct Words Percentage= ", (num_correct_words/len(preds)) * 100)
    corrected_file_path = pred_file_path.split("/")[:-1]
    corrected_file_path.append(CORRECTED_FILE_NAME + str(top_n) + TXT)

    corrected_file_path = "/".join(corrected_file_path)
    print("Words predicted incorrectly", words_predicted_incorrectly_list)
    print("Predictions stored here:", corrected_file_path)
    with open(corrected_file_path, "w") as txt_file:
        txt_file.write(f'Number of Correct Words = {num_correct_words}\n' )
        txt_file.write(f'Correct Words Percentage = {(num_correct_words/len(preds)) * 100}\n' )
        for line in corrected_words:
            txt_file.write(line + "\n")
        txt_file.write(f"\nWords Predicted Correctly\n")
        for line in words_predicted_correctly_list:
            txt_file.write(f"{line}\n")
        txt_file.write(f"\nWords Predicted Incorrectly\n")    
        for line in words_predicted_incorrectly_list:
            txt_file.write(f"{line}\n")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_path', help='path of predciton.txt')
    parser.add_argument('-v', '--vocab-path', help='path to the txt file containing the vocabulary')
    parser.add_argument('-t', '--top-n', default= 1, help='top1, top2, top3, top4 accuracy', type = int )
    
    args = parser.parse_args()
    vocabulary = get_vocabulary_set(args.vocab_path)
    correct_predictions(args.prediction_path, vocabulary, args.top_n)
