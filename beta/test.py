def find_most_negative_sum(lst):
  """
  This function finds the most negative sum of consecutive negative numbers in a list.

  Args:
      lst: A list of numbers.

  Returns:
      The most negative sum of consecutive negative numbers in the list, 
      or 0 if there are no negative numbers.
  """
  current_sum = 0
  most_negative_sum = 0
  for num in lst:
    if num < 0:
      current_sum += num
    else:
      most_negative_sum = max(most_negative_sum, current_sum)
      current_sum = 0
  return max(most_negative_sum, current_sum)

# Example usage
sample_list = [1, -2, -3, 4, -1, -2, -3, 5, -1]
most_negative_sum = find_most_negative_sum(sample_list)
print(most_negative_sum)  # Output: -8
