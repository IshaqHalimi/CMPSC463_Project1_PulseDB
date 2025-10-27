# kadane.py
# Kadane’s algorithm – find maximum subarray sum (used for activity detection)

def max_subarray(arr):
    if len(arr) == 0:
        return (0, -1, 0.0)
    best_sum = cur_sum = arr[0]
    start = 0
    best_range = (0, 0)

    for i in range(1, len(arr)):
        if cur_sum < 0:
            cur_sum = arr[i]
            start = i
        else:
            cur_sum += arr[i]
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_range = (start, i)

    return best_range[0], best_range[1], float(best_sum)
