import re


def zad_1(string: str) -> str:
    # new_str = re.findall(r'(\W+)', string)
    new_str = re.findall(r"[:;][-]?[)\-(<>]", string)
    emot = ''
    for e in new_str:
        emot += e
    new = string.lower()
    new = re.sub(r'[0-9]', '', new)
    new = re.sub(r'<.*?>', ' ', new)
    new = re.sub(r'[\W]', ' ', new)
    new = re.sub(' +', ' ', new)
    all = new + emot
    print(all)
