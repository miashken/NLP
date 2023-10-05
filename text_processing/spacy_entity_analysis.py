import spacy

MULTI_LANGUAGE_ENTITY_ANALYZER = "xx_ent_wiki_sm"

if not spacy.util.is_package(MULTI_LANGUAGE_ENTITY_ANALYZER):
    spacy.cli.download(MULTI_LANGUAGE_ENTITY_ANALYZER)

nlp_wk = spacy.load(MULTI_LANGUAGE_ENTITY_ANALYZER)

def get_addresses(text):
    """
    :param text (string): The text to analyze.
    :return list of addresses, and their location in the text (start and end indices)
    """
    doc = nlp_wk(text)

    locations = []

    for ent in doc.ents:
        if ent.label_ in ["LOC"]:
            locations.append([ent.text, ent.start, ent.end])
    
    return locations


if __name__ == "__main__":
    res = get_addresses("The top Democratic candidates in the unruly House race to \
            represent lower Manhattan and brownstone Brooklyn are scheduled to square \
            off Wednesday night in a key primetime debate on PIX11.")
    print(res)
