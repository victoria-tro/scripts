# BERT
print("Importing ML summarization packages")
from summarizer import Summarizer,TransformerSummarizer

# Text
body = '''
Silkie

The Silkie (also known as the Silky or Chinese silk chicken) is a breed of chicken named for its atypically fluffy plumage, which is said to feel like silk and satin. The breed has several other unusual qualities, such as black skin and bones, blue earlobes, and five toes on each foot, whereas most chickens only have four. They are often exhibited in poultry shows, and also appear in various colors. In addition to their distinctive physical characteristics, Silkies are well known for their calm, friendly temperament. It is among the most docile of poultry. Hens are also exceptionally broody, and care for young well. Although they are fair layers themselves, laying only about three eggs a week, they are commonly used to hatch eggs from other breeds and bird species due to their broody nature.

It is unknown exactly where or when these fowl with their singular combination of attributes first appeared, but the most well documented point of origin is ancient China. Other places in Southeast Asia have been named as possibilities, such as India and Java. The earliest surviving Western written account of Silkies comes from Marco Polo, who wrote of a "furry" chicken in the 13th century during his travels in Asia. In 1598, Ulisse Aldrovandi, a writer and naturalist at the University of Bologna, Italy, published a comprehensive treatise on chickens which is still read and admired today. In it, he mentions "wool-bearing chickens" and ones "clothed with hair like that of a black cat".

Silkies most likely made their way to the West via the Silk Route and maritime trade. The breed was recognized officially in North America with acceptance into the Standard of Perfection in 1874. Once Silkies became more common in the West, many myths were perpetuated about them. Early Dutch breeders told buyers they were the offspring of chickens and rabbits, while sideshows promoted them as having actual mammalian fur.

In the 21st century, Silkies are one of the most popular and ubiquitous ornamental breeds of chicken. They are often kept as ornamental fowl or pet chickens by backyard keepers, and are also commonly used to incubate and raise the offspring of other chickens and waterfowl like ducks, geese and game birds such as quail and pheasants.

Silkies are considered a bantam breed in some countries, but this varies according to region and many breed standards class them officially as large fowl; the bantam Silkie is actually a separate variety most of the time. Almost all North American strains of the breed are bantam-sized, but in Europe the standard-sized is the original version. However, even standard Silkies are relatively small chickens, with the males weighing only 1.8 kilograms (4 pounds), and females weighing 1.4 kg (3 lb). The American Standard of Perfection calls for males that are 1 kg (36 oz), and females that are 900 g (32 oz).

Silkie plumage was once unique among chicken breeds, however in recent years silkie feathering has been developed in several breeds, mostly notably the Chabo, where it is now standardised in Britain and the Netherlands. It has been compared to silk, and to fur. The overall result is a soft, fluffy appearance. Their feathers lack functioning barbicels, and are thus similar to down on other birds. This characteristic leaves Silkies unable to fly.

Silkies appear in two distinct varieties: bearded and non-bearded. Bearded Silkies have an extra muff of feathers under the beak area that covers the earlobes. They also are separated according to color. Colors of Silkie recognized for competitive showing include black, blue, buff, grey, partridge, and white. Alternative hues, such as cuckoo, lavender, red, and splash also exist. The standards of perfection call for all Silkies to have a small walnut-shaped comb, dark wattles, and turquoise-blue earlobes. In addition to these defining characteristics, Silkies have five toes on each foot. Other breeds which exhibit this rare trait include the Dorking, Faverolles, Houdan, and Sultan.

All Silkies have black or bluish skin, bones and grayish-black meat; they are in the group of Chinese fowls known by the Chinese language name of wu gu ji (烏骨雞), meaning 'black-boned chicken'. More specifically, the Silkie breed itself is named Taihe wu ji (泰和乌鸡), 'black-boned chicken from Taihe'. Other wu gu ji may not share characteristics of the Taihe breed, such as the mulberry comb, white fur, blue ears, and polydactyly.

Melanism which extends beyond the skin into an animal's connective tissue is a rare trait, and in chickens it is caused by fibromelanosis, which is a rare mutation believed to have begun in Asia. The Silkie and several other breeds descended from Asian stock possess the mutation. Disregarding color, the breed does not generally produce as much as the more common meat breeds of chicken.

Silkies lay a fair number of eggs, ranging from white to cream or light tan, but production is often interrupted due to their extreme tendency to go broody. A silkie hen can produce 100 eggs in an ideal year. Their capacity for incubation, which has been selectively bred out of most fowl bred especially for egg production, is often exploited by poultry keepers by allowing Silkies to raise the offspring of other birds.
'''

print("My text: \n", body)

# Load BERT model
print("BERT model: \n")
bert_model = Summarizer()
bert_summary = ''.join(bert_model(body, min_length=60))
print("BERT model summary: \n")
print(bert_summary)


# Load and execute GPT-2 model
print("\nGPT-2 model medium summary: \n")
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(body, min_length=60))
print(full)

# Load and execute GPT-2 model
print("\nGPT-2 model large summary: \n")
GPT2_model_large = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-large")
full_large = ''.join(GPT2_model_large(body, min_length=60))
print(full_large)