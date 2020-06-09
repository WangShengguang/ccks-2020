import pandas as pd


def format():
    data_df = pd.read_csv('./answer.txt')

    def func(text: str):
        if isinstance(text, str):
            ents = []
            for ent in set(text.split('\t')):
                if ent.startswith('<') or ent.startswith('"'):
                    ents.append(ent)
                else:
                    ents.append('"' + ent + '"')
                    print('"' + ent + '"')
            return '\t'.join(ents)
        else:
            print(text)
            return ""

    with open('./result.txt', 'w', encoding='utf-8') as f:
        for answer in data_df['answer']:
            line = func(answer)
            f.write(line + '\n')
    # data_df[['commit']].to_csv('./result.txt', encoding='utf_8_sig', index=False)


if __name__ == '__main__':
    format()
