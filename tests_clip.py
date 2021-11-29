from prediction.formain import get_clip
import pandas as pd
import itertools
import numpy as np
import shutil
import csv

lniveau1 = ['alcoholic drink','non-alcoholic drink','baby','unprocessed poultry','butchery minced meat','butchery prepared poultry cold cuts','meat substitute', 'dessert preparation','grocery','coffe tea infusion chicory','canned grocery','frozen grocery','milk','cream','egg','cheese','creamery fat','creamery dessert','prepared fish catering', 'unprocessed fish','smoked fish','bakery','pastry','fruit vegetable','pet store','exterior','home cleaning accesory','cleaning product','human hygiene','household laundry','household appliances','craft tools','supplies','decoration','food accessories','clothing','supplies','recreation and culture','multimedia','pharmaceutical']
lniveau2 = [['alcool','beer','wine'],['water','fruit juice','soda','flavored water','vegetable drink'],['baby food','baby milk','diapers','wipes','care','baby pharmaceutical'],['unprocessed poultry'],['minced meat'],['sausage','deli meats','prepared poultry'],['meat substitue'],['preparation for dessert'],['oil vinegar','spice condiment broth','flour','starch','pasta','rice','wheat','lentils','semolina','prepared dish','snacking','cookie','cereal','dried fruit','soup','sweets','chocolate','sauce','spreads'],['coffee','tea','infusion','chicory'],['canned vegetable','canned fruit','canned meat','canned fish'],['ice cream','french fries and similar'],['milk'],['cream'],['egg'],['cheese'],['butter','other fats'],['yoghurt','white cheese','dessert','compote'],['seafood'],['fish','shellfish'],['smoked fish'],['bread bun biscuit cake','brioche','cookies','cakes'],['pastries'],['fruits','vegetables'],['pet food','litter','hygiene','accessory'],['garden furniture','plants','insecticide','gardening tools'],['garbage can','sponge','glove','rags','tissues','paper towels','PQ','storage boxes and packaging','garbage bags','bucket','mops'],['aerosols','washing machine products','dishwasher products','detergent products','fuels'],['deodorants','cleasing gel','soap','hairbrush','cosmetics','makeup remover','hygienic protections','shaving','body hygiene','face care','dental hygiene','hair care','perfume','sex'],['household laundry'],['household appliances'],['craft tools'],['supplies'],['decoration'],['dishes','cooking paper','utensil'],['textile','leather goods'],['toys','book'],['CD','DVD','other multimedia'],['pharmaceutical']]
l_niveau1 = []
for i in lniveau2:
    index = lniveau2.index(i)
    for j in range(len(i)):
        l_niveau1.append(lniveau1[index])
l_niveau2 = list(itertools.chain(*lniveau2))

df_label = pd.DataFrame({'niv1':l_niveau1, 'niv2':l_niveau2})

df_1 = pd.read_csv("gitignore/auchan_product_legalInfo_sample.csv", encoding='utf-8')
df_2 = pd.read_csv("gitignore/auchan_product_media_sample.csv", encoding='utf-8')
df = pd.merge(df_1, df_2, how='right', left_on='ATPD_00101', right_on='cui')
df['description'] = df.ATPD_200133 + ' ' + df.product_label
df['image_name_MEDIASTEP'] = df['photo_id']
img_name = ['image_{}'.format(i) for i in range(1,len(df)+1)]
df.insert(1, "image_name", img_name, allow_duplicates=False)
df.insert(3, "label", 0, allow_duplicates=False)
df = df[['image_name_MEDIASTEP','image_name', 'description', 'label', 'media_url']]
df = df.dropna()

df.to_csv("gitignore/df.csv")
df_label.to_csv("gitignore/df_label.csv")

prediction_clip = df
prediction_clip = prediction_clip[['image_name','label']]
for i in prediction_clip.index:
    image_path = "gitignore/données/photos/{}.jpg".format(prediction_clip.image_name[i])
    label, score = get_clip(image_path, df_label, niv_tot=2)
    prediction_clip.label[i]=label
    print('image n°{}'.format(i))

prediction_clip.to_csv("gitignore/prediction_clip.csv")

