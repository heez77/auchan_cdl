# Projet Auchan

<p align="center">
       <img src="https://logo-marque.com/wp-content/uploads/2021/02/Auchan-Logo-1983-2015.png" width="300"/>
</p>

Le projet Auchan réalisé en collaboration avec Digital Lab a pour but de proposer une nouvelle classification des produits pour le service qualité à partir de données d'images et de textes.

## La structure du dépôt GitHub

<p> Ce dépôt GitHub recense le livrable final sans les données fournies par Auchan, en y accompagnant le processus d'installation sur une machine locale. (voir création d'une image Docker pour l'implémentation dans GCP). </p>
<p> Les configurations globales du code se trouvent dans le script "config.py". Les données sont placées dans les dossiers suivants : </p>
<pre>
> Data
  > Entrainement_bio
  > Entrainement_camemBERT
  > Entrainement_Dense
  > Entrainement_surgele
  > Predictions_bio
  > Predictions_classification
  > Predictions_surgele
</pre>
<p> Le dossier "Data" peut-être déplacé mais doit être reconfiguré dans le script "config.py". </p>

## Framework

à venir

## Requirements

Ci-dessous les requirements sur une machine locale.

[Python 3.8.10](https://www.python.org/downloads/release/python-3810/)

[Git](https://gitforwindows.org/)

[Pytorch 1.8.0 ou plus + torchvision](https://pytorch.org/get-started/previous-versions/)
<p> ou run </p>
<pre> pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html' </pre>

<p> (Windows seulement) Rust : https://www.rust-lang.org/tools/install </p>

<p> Fast BERT 1.9.9 </p>
<pre>
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
</pre>

CLIP
<pre>
pip install git+https://github.com/openai/CLIP.git
</pre>

EffDet

Télécharger [Visual Studio Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
<p> Cocher les cases suivantes et installer:
</p>
<p align="center">
       <img src="https://i.stack.imgur.com/pRpx0.png" width="800"/>
</p>

Packages dans 'requirements.txt'
<pre> pip install -r requirements.txt </pre>


run:
<pre>
import nltk
nltk.download('stopwords')
</pre>

## Object detection

Nous avons utilisé une méthode afin de déterminer si un produit est bio/non bio et surgelé/non surgelé à partir de reconnaissance d'indices sur l'image du produit. Le modèle utilisé est un EfficientDetB7 entraîné sur le dataset d'Auchan et qui permet de dessiner des bouding boxes autour d'une potentiel mention écrite "BIO" ou des logos bio :
<p align="center">
       <img src=https://agriculture.gouv.fr/sites/minagri/files/styles/affichage_pleine-page_790x435/public/logoab_eurofeuille_biologique.png?itok=yTSXiRzk width="100"/>
</p>
et des bouding boxes autour de logos surgelés :
<p align="center">
       <img src=https://cdn.discordapp.com/attachments/910086422889902100/931189223002886184/logo_thermo.png width="100"/>
       <img src=https://cdn.discordapp.com/attachments/910086422889902100/931189223250346104/logo_flocon.png width="100"/>
</p>
Les modèles entraînés sont disponibles sur le Drive partagé avec Auchan.

## Product classification

Le modèle baptisé "Dense" fusionne la classification proposée à la fois par le modèle [CLIP](https://github.com/openai/CLIP) (pour les données d'images) et à la fois par un modèle [CamemBERT](https://camembert-model.fr/) (pour les données textuelles) qui a été "fine-tuné". Les résultats sur les données de test donnent une précision >80% de classification d'un produit dans l'une des 79 catégories pré-définies. La méthode des seuils
<p> Le modèle CLIP a pour but de rattacher une image à un élément d'une liste de descriptions donnée. Cette liste de description ne dépend pas du modèle. Ainsi, nous pouvons utiliser le modèle entraîné sur pas moins de 40 millions de couple images/textes pour prédire la "distance" d'une image d'un produit aux 79 catégories prédéfnies par Auchan. </p>
<p> Le modèle CamemBERT a pour but de rattacher une description à un élément d'une liste de labels donnée. Nous avons fait du "transfert-learning" pour affiner le modèle spécifiquement aux données d'Auchan. Dans notre cas, les descriptions prises sont :
<ol>
  <li> les descriptions fournisseurs </li>
  <li> les descriptions produits </li>
</ol>
</p>
Les modèles entraînés sont disponibles sur le Drive partagé avec Auchan.

## Fonctionnement des algorithmes :

<p> Les scripts étant entièrement commentés, les différentes fonctions seront vues sans rentrer dans les détails. </p>

### CLIP :

Dans cette partie, seuls les fonctionnalités qui nous intéressent pour notre modèle seront détaillées. [CLIP](https://openai.com/blog/clip/) propose de faire de la prédiction 'zero-shot'. En proposant une liste de labels à CLIP, il peut directement donner le meilleur label parmi la liste pour une image donnée, sans ré-entrainement.
<p> Pour utiliser le modèle CLIP sur Python, il suffit de faire appel à la fonction clip.load, qui permet aussi de choisir quel vision transformer utiliser (un vision transformer encode une image ce qui facilite le traitement) </p>
<p> CLIP renvoie la liste des différents labels avec la probabilité que ce soit le meilleur label pour chaque image prédite. </p>

### CamemBERT :

Pour faire un modèle de classification multilabels basée sur le modèle [CamemBERT](https://github.com/utterworks/fast-bert) nous allons utiliser fast-bert, un package permettant de configurer le modèle CamemBERT pour l'entrainer avec ses propres labels.

#### Fine-tuning du modèle :

<p> Le fine-tuning consiste à ré-entrainer le modèle CamemBERT avec un learning rate très faible, ce qui fait que le modèle ne va que très peu modifier pour s'adapter aux types de textes présents dans le dataset sans perdre la qualité d'apprentissage du modèle de base. Pour entrainer ce modèle il faut créer un databunch, objet python chargé de référencer la localisation des différentes données, faire le preprocessing et paramétrer le choix du modèle utilisé. </p>
<pre>
databunch_lm = BertLMDataBunch.from_raw_corpus(
                        data_dir=DATA_PATH,
                        text_list=texts,
                        tokenizer='camembert-base',
                        batch_size_per_gpu=4,
                        max_seq_length=512,
                        multi_gpu=False,
                        model_type='camembert-base',
                        logger=logger)
</pre>
<p> Il faut ensuite créer un learner qui va s'entrainer sur le corpus entier du dataset. Ce sont les poids du learner à la fin de l'entrainement qui sont sauvegardées. Pour charger le modèle tout juste sauvegardé, il suffit de connaître le chemin d'accès aux poids et l'utiliser comme fine-tuned weight path à la création du modèle de classification BERT. </p>
<p> L'entraînement pour la classification BERT est similaire, il faut aussi créer un databunch, mais cette fois-ci pour la classification avec un dataset d'entraînement, de validation et de test, la liste des labels ainsi que le type de modèle utilisé (camemBERT pour nous). </p>
<pre>
databunch = BertDataBunch(DATA_PATH, DATA_PATH,
                          tokenizer='camembert-base',
                          train_file='train.csv',   
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=labels,
                          batch_size_per_gpu=4,
                          max_seq_length=512,
                          multi_gpu=False,
                          multi_label=True,
                          model_type='camembert-base')
</pre>
<p> Ensuite il faut créer un learner, qui va pouvoir récupérer un modèle camemBERT fine tuned et les autres infos nécessaires. </p>
<pre>
learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path=MODEL_PATH,
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR_BERT,
						finetuned_wgts_path=WGTS_PATH,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=True,
						multi_label=True,
						logging_steps=50)
</pre>
<p> Pour la prédiction, il suffit de charger le modèle en chargeant le dossier où le modèle s'enregistre. </p>

### EfficientDet :

<p> EfficientDet fonctionne sur le même principe que les modèles précédents, il faut d'abord créer un objet Python servant de référenceur de dataset puis un learner. Ici, le dataset est créé sur Python avec la fonction DatasetAdaptor qui se charge de faire le preprocessing des images et des annotations. </p>
<pre>
class DatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe): 
        self.images_dir_path = images_dir_path
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)
    class_labels = []
    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = PIL.Image.open(self.images_dir_path + image_name)
        pascal_bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values # Récupération des rectangles 
        class_labels = self.annotations_df[self.annotations_df.image == image_name][
            ["class"]].values.tolist() # Récupération des labels pour chaque rectangle 
        c_l = np.zeros(len(class_labels))
        for i,cl in enumerate(class_labels):
            c_l[i] = dico[cl[0]]

        return image, pascal_bboxes, c_l, index
    def dict_for_path(self):
        dico={}
        for index in range(len(self.images)):
            image_name = self.images[index]
            dico['image_{}.jpeg'.format(index)]=image_name
        return dico
</pre>
<p> Ensuite, il faut créer un itérateur qui permet au learner d'accéder plus facilement aux données d'entraînement et de validation grâce à la fonction EfficientDetDataModule. </p>
<pre>
class EfficientDetDataModule(LightningDataModule):
    
    def __init__(self,
                train_dataset_adaptor,
                validation_dataset_adaptor,
                train_transforms=get_train_transforms(target_img_size=512),
                valid_transforms=get_valid_transforms(target_img_size=512),
                num_workers=4,
                batch_size=1):
        
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids
</pre>

<p> On créer ensuite le modèle avec la fonction EfficientDetModel qui prend en argument le nombre de labels dans le dataset. </p>
<pre>
photo
</pre>
<p> Pour entraîner, il faut créer un objet Trainer qui créer un environnement pour l'entraînement avec le nombre d'epochs et l'enregistrement des logs. </p>

<p> En fournissant une liste de photos à l'algorithme, il renvoie un fichier csv comprenant le nom de l'image et si il a détecté un label ou non. Il est aussi possible de générer une nouvelle image avec les rectangles prédis par le modèle. </p>

## Entraînement des modèles :

### Efficient Det :

Les démarches pour le surgelé et le bio sont identiques, nous ne traiterons que le cas de la détection de produits bio.

#### Preprocessing :

<p> Ici, pas besoin d'avoir un dataset d'entraînement comportant des produits bios et non bios, le modèle va uniquement apprendre à repérer les motifs correspondant aux produits bios. Pour ce faire il nous faut une image et les annotations correspondantes : </p>

<p> Elles correspondent aux rectangles encadrant un motif bio ainsi que le bon label : </p>
liste des labels pour les produits bios :

<p> La variable labels peut être modifié en fonction des labels identifiés. </p>
<pre> capture ecran variable labels </pre>

L'algorithme [EfficientDet](https://github.com/xuannianz/EfficientDet) fonctionne avec des annotations sous forme de fichier csv. Un fichier annotation est présent pour chaque produit qui comporte les coordonnnées du rectangle (x1, x2, y1, y2) de la bouding box et le nom du label correspondant sous format [Pascal VOC](https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5). Comme beaucoup de logiciels d'annotations d'images ne renvoient pas les annotations sous format csv mais sous format xml, la fonction convert_xml convertit tous les fichiers xml en fichiers csv.
<p> Il faut fournir au modèle l'ensemble des images des produits et les annotations sous format xml dans le répertoire '/PATH_TO_FOLDER/Donnees/Entrainement_Bio/' </p>

#### Script d'exécution :

En exécutant le script python3 entrainement_bio.py un nouveau modèle est entrainé avec les nouvelles données et enregistré dans le répertoire 'PATH_TO_FOLDER/Model/Bio/'.

### Classification :

#### Preprocessing :

<p> CamemBERT s'entraîne uniquement sur des données textuelles, nous allons lui fournir les descriptions fournisseurs et produits ainsi que le label correspondant. Il n'est pas nécessaire de fournir la liste des labels le modèle va uniquement s'entrainer sur les données présentes dans le dataset. </p>
<p> Format du dataset d'entrée : csv </p>
<p> Colonnes : A remplir </p>

#### Script d'exécution :

<pre> python3 entrainement_camemBERT.py </pre>

Il est possible d'ajouter l'argument --tuning à cette ligne de commande qui est un booléen (True ou False) permettant de choisir de faire ou non du fine-tuning sur le modèle CamemBERT. Par défaut l'argument tuning est à False. Exemple :
<pre> python3 entrainement_camemBERT.py --tuning True </pre>
<p> Nouveau modèle sauvegardé dans le répertoire : 'PATH_TO_FOLDER/Model/Classification/'. (supprimer l'autre loss) </p>

## Tensorboard :

Pour chaque entraînement effectué, les logs des entraînements sont sauvegardées et peuvent être visualisées avec Tensorboard.

### Ouverture de Tensorboard : 
Si Tensoboard n'est pas installé, faire
<pre> pip install tensorboard </pre>
Puis entrer dans une ligne de commande Windows, après avoir activé l'environnement virtuel (s'il y en a un) :
<pre>  tensorboard --logdir 'PATH/TO/logs/fit' </pre>

### Ouverture de Tensorboard sous Visual Studio Code :
<ol>
  <li> Installer l'extension Tensorboard. </li>
  <li> Bien vérifier qu'on est dans le bon répertoire. </li>
  <li> Dans la barre de recherche : CTRL + SHIFT + P </li>
  <li> Entrer : Tensorboard </li>
  <li> Faire Entrée </li>
</ol>

Une nouvelle page Tensorboard s'ouvre et nous pouvons voir des résultats :

<p align="center">
       <img src="https://cdn.discordapp.com/attachments/910086422889902100/931484866409795635/unknown.png" width="800"/>
</p>

<p> A gauche, nous avons accès à tous les entraînements par type de modèle (dossiers) puis par version en cochant ou décochant les cases. Il est ainsi possible de comparer diférentes versions d'un même modèle. Seul CLIP n'est pas disponible sur Tensorboard car il ne s'entraîne pas. </p>
<p> Tensorboard permet de visualiser les différentes metrics évaluées au cours de l'entraînement, la loss ainsi que la complexité (utile pour déterminer la configuration nécessaire pour avoir un temps d'entraînement raisonnable). </p>

## Prédiction :

### Prédiction classification - méthode des seuils

run "prediction_classification.py"

### Prédiction classification - méthode "Dense"

run "crea_csv_dense.py" puis "dense.py"

### Prédiction object-detection

run "prediction_bio.py" ou "prediction_surgele.py"

## GCP :

<p> Le sigle GCP signifie Google Cloud Platform. Cette plateforme Cloud contient notamment l'outil Vertex AI spécialisé dans le Machine Learning. </p>

<p> Pour ce projet, nous avons :
<ol>
  <li> Placé nos données dans un bucket </li>
  <li> Lancé plusieurs Custom Jobs correspondant à des tentatives d'entraînement de l'algorithme CamemBERT </li>
</ol>
</p>

<p> Pour lancer le custom jobs, il faut au préalable installer Cloud SDK et créer un package Python dans lequel est présent le code (dans un fichier task.py). Nous nous sommes cependant heurtés rapidement à des problèmes de permission. En particulier, pour récupérer les données des buckets, il faut installer la librairie Google-cloud-storage et utiliser la fonction storage : </p>
<pre> from google-cloud import storage </pre>
<p> Cependant, l'utilisation de ces fonctions nécessitent des droits que nous n'avions pas. </p>
