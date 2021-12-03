import pandas as pd
import transformers as ppb
import numpy as np
import torch
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from transformers import CamembertTokenizer, CamembertModel
from formain import get_clip

#url="https://raw.githubusercontent.com/nlpinfrench/nlpinfrench.github.io/master/source/labeled_data.csv"
# df_1 = pd.read_csv("gitignore/auchan_product_legalInfo_sample.csv", encoding='utf-8')
# df_2 = pd.read_csv("gitignore/auchan_product_media_sample.csv", encoding='utf-8')

# lniveau1 = ['alcoholic drink','non-alcoholic drink','baby','unprocessed poultry','butchery minced meat','butchery prepared poultry cold cuts','meat substitute', 'dessert preparation','grocery','coffe tea infusion chicory','canned grocery','frozen grocery','milk','cream','egg','cheese','creamery fat','creamery dessert','prepared fish catering', 'unprocessed fish','smoked fish','bakery','pastry','fruit vegetable','pet store','exterior','home cleaning accesory','cleaning product','human hygiene','household laundry','household appliances','craft tools','supplies','decoration','food accessories','clothing','supplies','recreation and culture','multimedia','pharmaceutical']
# lniveau2 = [['alcool','beer','wine'],['water','fruit juice','soda','flavored water','vegetable drink'],['baby food','baby milk','diapers','wipes','care','baby pharmaceutical'],['unprocessed poultry'],['minced meat'],['sausage','deli meats','prepared poultry'],['meat substitue'],['preparation for dessert'],['oil vinegar','spice condiment broth','flour','starch','pasta','rice','wheat','lentils','semolina','prepared dish','snacking','cookie','cereal','dried fruit','soup','sweets','chocolate','sauce','spreads'],['coffee','tea','infusion','chicory'],['canned vegetable','canned fruit','canned meat','canned fish'],['ice cream','french fries and similar'],['milk'],['cream'],['egg'],['cheese'],['butter','other fats'],['yoghurt','white cheese','dessert','compote'],['seafood'],['fish','shellfish'],['smoked fish'],['bread bun biscuit cake','brioche','cookies','cakes'],['pastries'],['fruits','vegetables'],['pet food','litter','hygiene','accessory'],['garden furniture','plants','insecticide','gardening tools'],['garbage can','sponge','glove','rags','tissues','paper towels','PQ','storage boxes and packaging','garbage bags','bucket','mops'],['aerosols','washing machine products','dishwasher products','detergent products','fuels'],['deodorants','cleasing gel','soap','hairbrush','cosmetics','makeup remover','hygienic protections','shaving','body hygiene','face care','dental hygiene','hair care','perfume','sex'],['household laundry'],['household appliances'],['craft tools'],['supplies'],['decoration'],['dishes','cooking paper','utensil'],['textile','leather goods'],['toys','book'],['CD','DVD','other multimedia'],['pharmaceutical']]
# l_niveau1 = []
# for i in lniveau2:
#     index = lniveau2.index(i)
#     for j in range(len(i)):
#         l_niveau1.append(lniveau1[index])
# l_niveau2 = list(itertools.chain(*lniveau2))

# df_label = pd.DataFrame({'niv1':l_niveau1, 'niv2':l_niveau2})

# df = pd.merge(df_1, df_2, how='right', left_on='ATPD_00101', right_on='cui')
# df['description'] = df.ATPD_200133 + ' ' + df.product_label
# df = df[['photo_id','description']]
# df = df.dropna()
# # Report the number of sentences.
# print(df.head(5))
# print('Number of sentences: {:,}\n'.format(df.shape[0]))
# remove unuseful columns
# df = df[["review","temps"]]
# Display 5 random rows from the data.
# print(df.head(5))


df = pd.read_csv("gitignore/df.csv")

BATCH_SIZE = 4
NUMBER_COMPLETE_BATCH = BATCH_SIZE // len(df)
REST_BATCH = BATCH_SIZE % len(df)

# load model, tokenizer and weights
camembert, tokenizer, weights = (ppb.CamembertModel, ppb.CamembertTokenizer, 'camembert-base')
print(tokenizer)
print(weights)

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
camembert = CamembertModel.from_pretrained("camembert-base")
print("tokenizer = ", tokenizer)
# Load pretrained model/tokenizer
tokenizer = tokenizer.from_pretrained(weights)
model = camembert.from_pretrained(weights)

# see if there are length > 512
max_len = 0
for i,sent in enumerate(df["description"]):
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    if len(input_ids) > 512:
        print("annoying review at", i,"with length",
              len(input_ids))
    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)

tokenized = df['description'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

for complete_batch in range(NUMBER_COMPLETE_BATCH):
    print(complete_batch)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized[5*complete_batch:5*complete_batch+4].values])
    attention_mask = np.where(padded != 0, 1, 0)
    print('for complete_batch n°{} we have padded :\n{}'.format(complete_batch,padded))

print('OK Padded')

device = "cuda" if torch.cuda.is_available() else "cpu"

print('device = ', device)

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)
 
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()
labels = df.label_niv2
print(labels)

train_features, test_features = train_test_split(features)

# Grid search
parameters = {'C': np.linspace(0.0001, 100, 20)}
grid_search = GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs'), parameters)
grid_search.fit(train_features, labels)

print('best parameters: ', grid_search.best_params_)
print('best scrore: ', grid_search.best_score_)