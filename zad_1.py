import re


def zad_1_a():
    string = 'Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku'
    pattern = r'[0-9]'
    new_str = re.sub(pattern, '', string)
    new_str = re.sub(' +', ' ', new_str)
    print(new_str)


def zad_1_b():
    string = '<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a></p></div>'
    pattern = r'<.*?>'
    new_str = re.sub(pattern, '', string)
    print(new_str)


def zad_1_c():
    string = 'Lorem ipsum dolor sit amet, consectetur; adipiscing elit. Sed eget mattis sem. Mauris egestas erat quam, ' \
             'ut faucibus eros congue et. In blandit, mi eu porta; lobortis, tortor nisl facilisis leo, at tristique ' \
             'augue risus eu risus.'
    pattern = r'[\W]'
    new_str = re.sub(pattern, ' ', string)
    print(new_str)
