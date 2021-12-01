from formain import get_clip
import pandas as pd
import itertools
import numpy as np
import os

lniveau1_eng = ['alcoholic drink','non-alcoholic drink','baby','unprocessed poultry','butchery minced meat','butchery prepared poultry cold cuts','meat substitute', 'dessert preparation','grocery','coffe tea infusion chicory','canned grocery','frozen grocery','milk','cream','egg','cheese','creamery fat','creamery dessert','prepared fish catering', 'unprocessed fish','smoked fish','bakery','pastry','fruit vegetable','pet store','exterior','home cleaning accesory','cleaning product','human hygiene','household laundry','household appliances','craft tools','supplies','decoration','food accessories','clothing','supplies','recreation and culture','multimedia','pharmaceutical']
lniveau2_eng = [['alcool','beer','wine'],['water','fruit juice','soda','flavored water','vegetable drink'],['baby food','baby milk','diapers','wipes','care','baby pharmaceutical'],['unprocessed poultry'],['minced meat'],['sausage','deli meats','prepared poultry'],['meat substitue'],['preparation for dessert'],['oil vinegar','spice condiment broth','flour','starch','pasta','rice','wheat','lentils','semolina','prepared dish','snacking','cookie','cereal','dried fruit','soup','sweets','chocolate','sauce','spreads'],['coffee','tea','infusion','chicory'],['canned vegetable','canned fruit','canned meat','canned fish'],['ice cream','french fries and similar'],['milk'],['cream'],['egg'],['cheese'],['butter','other fats'],['yoghurt','white cheese','dessert','compote'],['seafood'],['fish','shellfish'],['smoked fish'],['bread bun biscuit cake','Bakery brioche','Bakery cookies','Bakery cakes'],['pastries'],['fruits','vegetables'],['pet food','litter','hygiene','accessory'],['garden furniture','plants','insecticide','gardening tools'],['garbage can','sponge','glove','rags','tissues','paper towels','PQ','storage boxes and packaging','garbage bags','bucket','mops'],['aerosols','washing machine products','dishwasher products','detergent products','fuels'],['deodorants','cleasing gel','soap','hairbrush','cosmetics','makeup remover','hygienic protections','shaving','body hygiene','face care','dental hygiene','hair care','perfume','sex'],['household laundry'],['household appliances'],['craft tools'],['supplies'],['decoration'],['dishes','cooking paper','utensil'],['textile','leather goods'],['toys','book'],['CD','DVD','other multimedia'],['pharmaceutical']]
l_niveau1_eng = []
for i in lniveau2_eng:
    index = lniveau2_eng.index(i)
    for j in range(len(i)):
        l_niveau1_eng.append(lniveau1_eng[index])
l_niveau2_eng = list(itertools.chain(*lniveau2_eng))

lniveau1_fr = ['Boissons alcoolisées','Boissons sans alcool','Bébé','Volaille non prép','Boucherie - viandes hachées','Boucherie/Volaille préparé/Charcuterie','Boucherie - succédanés de viande',"Epicerie - avec mode d'emploi","Epicerie","Epicerie - sans valeurs nutritionnelles","Epicerie - Conserves (boîtes et bocaux)","Epicerie surgelée","Crémerie - laits","Crémerie - crèmes","Crémerie - oeufs","Crémerie - fromages","Crèmeries - matières grasses","Cremerie desserts","Poissonnerie traiteur /préparé","Poissonnerie non préparé","Poissonerie fumaison","Boulangerie","pâtisserie","Fruits et légumes","Animalerie","Exterieur","Entretien Maison - accessoires","Entretien produits /détergents","Hygiène humaine","Linge de maison","Electroménager","Outils de bricolage","Décoration","Accessoires alimentaires","Habits","Fourniture","Loisirs et culture","Multimedia","pharmaceutique"]
lniveau2_fr = [['Alcools','Bieres','Vins'],['Eau','Jus de fruits','Sodas','eaux aromatises','Boissons vegetales'],['Aliments bebe','Laits bebe','Couches','Lingettes','Soin','pharmaceutique pour bebe'],['Volaille non prep'],['viandes hachees'],['Saucisses','Charcuterie','Volaille prep'],['succedanes de viande'],['Preparations pour dessert'],['Huiles et vinaigres','Epices et condiments bouillons','Farines','Fecules','Pates','Riz','Ble','Lentilles','Semoules','Plats prepares','snacking','Biscuiterie','Cereales','Fruits secs','soupes','confiseries','chocolats','sauces','tartinables'],['cafe','the','infusions','chicorees'],['Conserves de legumes','Conserves de fruits','Conserves de viandes','Conserves de poisson'],['glaces','frites et assimiles'],['Laits'],['Cremes'],['Oeufs'],['Fromages'],['Beurres','Autres matieres grasses'],['yaourts','fromages blancs','desserts','compotes'],['Traiteur mer'],['Poisson','Crustaces'],['Fumaison poisson'],['pains brioches  biscuits  gateaux','Boulangerie Brioches','Boulangerie biscuits','Boulangerie gateaux'],['patisseries'],['Fruits','Legumes'],['Nourriture pour animal','Litiere','Hygiene','Accessoire'],['Mobilier jardin','Plantes','insecticide','Outils de jardinage'],['poubelle','eponge','gant','chiffons','Mouchoirs','Essuie tout','PQ','emballage','Sacs poubelles','seaux','serpilleres'],['Aerosols','produits machine a laver','produits lave vaiselle','produits detergents','combustibles'],['Deodorants','Gel nettoyant','savon','brosse a cheveux','Cosmetiques (Maquillage)','Demaquillant','Protections hygeniques','Rasage','hygiene corporelle','Soin du visage','Hygiene dentaire','Soin des cheveux','Parfum','sexe (lubrifiant + preservatif)'],['Linge de maison'],['Electromenager'],['Outils de bricolage'],['Decoration'],['Vaisselle','Papiers cuissons','Ustensile'],['Textile','Maroquinerie'],['Fourniture'],['Jouets','Livre'],['CD','DVD','autre multimedia'],['pharmaceutique']]
l_niveau1_fr = []
for i in lniveau2_fr:
    index = lniveau2_fr.index(i)
    for j in range(len(i)):
        l_niveau1_fr.append(lniveau1_fr[index])
l_niveau2_fr = list(itertools.chain(*lniveau2_fr))

df_label = pd.DataFrame({'niv1_eng':l_niveau1_eng, 'niv1_fr':l_niveau1_fr, 'niv2_eng':l_niveau2_eng, 'niv2_fr':l_niveau2_fr})
df_label.to_csv("gitignore/df_label.csv")

df_1 = pd.read_csv("gitignore/auchan_product_legalInfo_sample.csv", encoding='utf-8')
df_2 = pd.read_csv("gitignore/auchan_product_media_sample.csv", encoding='utf-8')
df = pd.merge(df_1, df_2, how='right', left_on='ATPD_00101', right_on='cui')
df['description'] = df.ATPD_200133 + ' ' + df.product_label
df['image_name_MEDIASTEP'] = df['photo_id']
img_name = ['image_{}'.format(i) for i in range(0,len(df))]
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

