def get_distinct_chars(filename):
    distinct_chars = set()
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            distinct_chars.update(line)
    
    return ''.join(sorted(distinct_chars))

# Example usage
filename = r'D:\Workspace\python_code\ImageGenerations\texts\data.txt'
result = get_distinct_chars(filename)
print(f"Distinct characters: {result}")