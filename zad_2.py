import re


def zad_2():
    string = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed #texting eget mattis sem. Mauris #frasista' \
             ' egestas erat #tweetext quam, ut faucibus eros #frasier congue et. In blandit, mi eu porta lobortis, ' \
             'tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus.'
    # pattern = r'#(\w+)'
    pattern = r'#'
    new_str = re.findall(pattern, string)
    print(new_str)
