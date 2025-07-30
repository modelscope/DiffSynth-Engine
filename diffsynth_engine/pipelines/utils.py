def accumulate(result, new_item):
    if result is None:
        return new_item
    for i, item in enumerate(new_item):
        result[i] += item
    return result
