from collections import Counter

def sum_counts(inList):
  print(sum(inList))

def count_numbers(inList):
  number_counts = Counter(inList)
  # If you want to see the counts in ascending order by the number
  sorted_counts = dict(sorted(number_counts.items()))

  # To print the counts
  for number, count in sorted_counts.items():
    print(f"{number}: {count} times")

def sort_list(inList):
  return sorted(inList)

winnerB_correct_confidences = [
  41,44,47,46,48,55,51,46,38,80,43,38,46,28,44,34,47,62,68,35,54,36,40,27,67,70,53,67,
  43,29,75,41,43,61,55,31,74,16,22,36,55,40,57,71,41,42,37,65,56,58,33,74,58,47,53,52,
  38,34,55,24,60,53,50,42,26,22,62,65,16,65,40,38,48,50,57,42,48,25,46,50,57,51,62,33,
  40,60,39,52,33,44,45,72,57,24,69,50,25,33,56,55,47,52,24,52,77,53,35,22,25,31,46,67,
  58,34,42,68,40,56,49,82,33,37,54,48,52,46,30,41,34,33,54,35,55,36,36,50,43,46,51,34,
  24,25,63,71,30,49,35,55,39,31,30,35,20,52,62,49,21,55,42,59,32,41,30,48,64,29,19,36,
  28,37,31,51,41,48,45,53,58,66,73,57,58,60,54,63,41,28,34,41,46,51,20,38,31,41,78,28,
  42,41,27,50,44,31,49,50,30,42,45,32,58,33,62,45,38,50,32,47,46,71,21,55,49,78,46,34,
  47,44,63,23,35,30,55,51,48,41,68,37,58,26,43,49,36,46,50,68,43,48,56,46,57,53,36,60,
  54,47,41,40,37,41,62,36,27,37,47,49,24,36,33,26,42,40,30,30,36,46,41,51,38,46,30,38,
  55,62,37,28,61,63,49,35,76,54,48,62,49,64,47,38,59,37,52,79,50,50,41,36,31,34,51,54,
  54,39,34,63,48,72,20,24,33,43,33,68,62,41,61,45,16,16,63,39,61,47,76,61,23,29,46,30,
  42,55,51,54,78,37,42,20,63,21,34,35,54,55,53,78,18,49,35,56,58,25,51,25,55,60,68,80,
  45,40,62,67,56,49,37,49,52,51,56,56,18,31,56,47,70,46,46,35,42,42,68,38,56,28,46,43,
  28,44,65,25,72,64,37,56,38,65,71,43,38,49,66,44,46,34,38,29,35,42,34,36,36,38,24,36,
  49,48,22,55,58,49,62,50,39,66,38,61,39,33,59,45,34,44,60,37,38,32,22,78,78,67,30,62,
  37,29,35,65,48,25,29,31,25,17,22,70,59,35,47,22,16,45,20,39,61,48,62,38,27,55,69,26,
  50,32
]
winnerB_incorrect_confidences = [
  33,61,55,48,55,62,38,28,68,64,52,51,17,56,35,43,62,46,57,57,57,50,40,30,69,27,42,64,
  36,36,37,43,19,45,49,69,47,70,25,57,57,32,42,60,76,22,71,44,31,22,58,15,57,24,75,67,
  33,50,39,71,59,45,46,59,52,64,41,68,49,62,40,47,54,79,22,32,42,15,34,63,73,32,23,35,
  63,48,22,41,58,49,46,51,46,25,39,58,24,62,38,53,37,41,80,28,57,80,50,53,34,40,62,48,
  36,48,23,45,25,56,37,40,51,43,40,31,71,34,65,28,26,60,52,25,31,26,41,41,19,62,35,29,
  49,18,63,31,51,59,72,59,44,27,38,29,55,60,45,40,49,49,35,74,31,24,27,53,59,59,27,28,
  54,53,16,51,40,45,38,47,62,48,41,32,32,67,49,23,47,61,46,52,51,63,70,41,37,31,51,37,
  71,68,62,47,48,43,73,73,44,30,44,54,52,42,34,57,68,46,72,74,55,73,36,55,56,47,43,64,
  36,44,50,59,27,37,47,36,33,37,37,62,27,35,62,59,35,18,27,40,64,38,38,72,50,35,71,60,
  48,43,40,67,38,26,26,60,51,32,37,34,32,71,32,55,35,74,36,50,29,37,56,48,50,71,52,60,
  34,22,46,40,18,44,60,57,31,45,58,44,61,28,25,25,46,40,53,54,75,26,48,42,37,54,41,28,
  44,18,57,30,67
]

# sum_counts(winnerR_outcomes)
print('winnerB_correct_confidences len:',len(winnerB_correct_confidences))
print('winnerB_correct_confidences percent:', sum(winnerB_correct_confidences) / len(winnerB_correct_confidences))
print('winnerB_correct_confidences min:',min(winnerB_correct_confidences))
print('winnerB_correct_confidences max:',max(winnerB_correct_confidences))
print('winnerB_correct_confidences mode:',max(set(winnerB_correct_confidences), key=winnerB_correct_confidences.count))
print('winnerB_incorrect_confidences len:',len(winnerB_incorrect_confidences))
print('winnerB_incorrect_confidences percent:', sum(winnerB_incorrect_confidences) / len(winnerB_incorrect_confidences))
print('winnerB_incorrect_confidences min:',min(winnerB_incorrect_confidences))
print('winnerB_incorrect_confidences max:',max(winnerB_incorrect_confidences))
print('winnerB_incorrect_confidences mode:',max(set(winnerB_incorrect_confidences), key=winnerB_incorrect_confidences.count))


