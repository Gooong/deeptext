from deeptext.utils.csv import read_csv


def read_data(data_path):
    rows = read_csv(data_path)

    row_num = 0

    tokens = []
    labels = []
    for row in rows:
        if row_num % 2 == 0:
            tokens.append(row)
        else:
            labels.append(row)
        row_num += 1

    return tokens, labels        
