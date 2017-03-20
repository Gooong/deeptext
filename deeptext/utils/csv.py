import unicodecsv as csv


def read_csv(filename):
    rows = []
    with open(filename, 'r') as f:
        for row in csv.reader(f, encoding='utf-8'):
            rows.append(row)
    return rows
