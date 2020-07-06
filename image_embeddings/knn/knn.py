import pyarrow.parquet as pq

from dataclasses import dataclass
from IPython.display import Image, display
from ipywidgets import widgets, HBox, VBox
import faiss
import numpy as np

def read_embeddings(path):
  emb = pq.read_table(path).to_pandas()
  id_to_name = {k:v.decode("utf-8") for k,v in enumerate(list(emb["image_name"]))}
  name_to_id = {v:k for k,v in id_to_name.items()}
  embgood = np.stack(emb["embedding"].to_numpy())
  return [id_to_name, name_to_id, embgood]

def build_index(emb):
  d = emb.shape[1]
  xb = emb
  index = faiss.IndexFlatIP(d)
  index.add(xb)
  return index

def search(index, id_to_name, emb, k=5):
  D, I = index.search(np.expand_dims(emb, 0), k)     # actual search
  return list(zip(D[0], [id_to_name[x] for x in I[0]]))

def display_picture(image_path, image_name):
  display(Image(filename=f"{image_path}/{image_name}.jpeg"))

def display_results(image_path, results):
  hbox = HBox([VBox([widgets.Label(f"{distance:.2f} {image_name}"), widgets.Image(value=open(f"{image_path}/{image_name}.jpeg", 'rb').read())]) for distance, image_name in results])
  display(hbox)



