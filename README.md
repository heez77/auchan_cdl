# Projet Auchan

<p align="center">
       <img src="https://www.auchan.fr/content-renderer/sav_20211221-05/images/auchan-logo-desktop.svg">
</p>

Le projet Auchan réalisé en collaboration avec Digital Lab a pour but de proposer une nouvelle classification des produits pour le service qualité à partir de données d'images et de textes.

## La structure du dépôt GitHub

Ce dépôt GitHub recense le livrable final sans les données fournies par Auchan, en y accompagnant le processus d'installation sur une machine locale. (voir création d'une image Docker pour l'implémentation dans GCP).

## Framework

à venir

## Requirements

Ci-dessous la liste d'instructions pour une installation sur une machine locale.

<pre>
Python 3.8.10

Pytorch 1.8.0 or higher + torchvision, available here: https://pytorch.org/get-started/previous-versions/
or just run 'pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html'

packages in requirements.txt (run 'python -m pip install -r - requirements.txt')

(Windows only) Rust: https://www.rust-lang.org/tools/install

run:
import nltk
nltk.download()
</pre>

## Object detection

## Modèle "Dense"

Le modèle baptisé "Dense" fusionne la classification proposée à la fois par le modèle CLIP (pour les données d'images) et à la fois par un modèle CamemBERT (pour les données textuelles) qui a été "fine-tunné". Les résultats sur les données de test donnent une précision >80% de classification d'un produit dans l'une des 79 catégories pré-définies.
