def read_file_and_count_elements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    lists = []
    labels = []

    for i in range(0, len(lines), 2):
        # 리스트를 읽고 파싱
        lst = eval(lines[i].strip())
        lists.append(lst)
        
        # 라벨을 읽고 저장
        label = int(lines[i + 1].strip())
        labels.append(label)
    
    # 리스트의 항목 개수와 라벨 출력
    for lst, label in zip(lists, labels):
        count = sum(len(sublist) for sublist in lst)
        print(f'\n{label}\n{count}')

# 파일명에 맞게 파일을 읽어들임
filename = '[HRS]MI_four_1.txt'
read_file_and_count_elements(filename)
