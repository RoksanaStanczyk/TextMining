import re


def zad_3():
    string = 'To :) są;) różne emotki 123 ;( :> :< ;< :-)'
    pattern = r'(\W+\s)'
    new_str = re.findall(pattern, string)
    print(new_str)
