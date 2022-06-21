import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer

text = '''
Mole er en meksikansk kakaobasert saus, for anvendelse til fremfor alt kyllingretter og enchiladas. 
Mole lages med malt kakao, minst syv sorters chili, sesamfrø og hønsebuljong. Det finnes et utall ulike oppskrifter med hovedgruppene svart-, rød- og grønnmole. 
Svartmole er typisk for Oaxaca mens rødmole eller mole poblano (fra Puebla) er den mest populære i dagens Mexico.
'''

'''
Utregning av Liks:
Liks = X + Y
X = Antall ord/Antall setninger
Y = Antall ord over 6/Antall ord * 100
'''
def liks(text):
    liks_value = 0
    words_over_6 = 0
    number_of_sentences = len(sent_tokenize(text))

    # Tokenize text
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    number_of_tokens = len(tokens)

    for token in tokens:
        if len(token) > 6:
            words_over_6 += 1

    # Calculate Liks
    words_per_sent = number_of_tokens/number_of_sentences
    long_words_average = words_over_6/number_of_tokens * 100
    liks_value = words_per_sent + long_words_average

    return int(liks_value)
