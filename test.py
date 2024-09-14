file1 = r'D:\Downloads\cinamonai\test.txt'
file2 = r'D:\Downloads\vnondb\test.txt'
merged_file = r'D:\Downloads\vnondb\label_test.txt'

with open(file1, 'r',encoding='utf8') as f1, open(file2, 'r',encoding='utf8') as f2, open(merged_file, 'w',encoding='utf8') as mf:
    mf.write(f1.read())
    mf.write(f2.read())
