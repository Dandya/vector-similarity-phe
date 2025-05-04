from facenet_pytorch import MTCNN, InceptionResnetV1
from phe import paillier
from sys import argv
from math import sqrt
from pathlib import Path
import torch
from PIL import Image
import json
import typing
import os

class Embedding:
	def __init__(self, data: list, is_enc: bool = False, encrypt_data: dict = {}):
		if not is_enc:
			if len(data) == 0:
				raise Exception("data must have at least 1 element")
			self.data = data
			self.is_enc = False
		else:
			self.data = []
			for i in range(len(data)):
				self.data.append(paillier.EncryptedNumber(encrypt_data["pub"], data[i][0], data[i][1]))
			self.is_enc = True

	def normalize(self):
		l = 0
		for val in self.data:
			l += val ** 2
		l = sqrt(l)
		for i in range(len(self.data)):
			self.data[i] = self.data[i] / l

	def encrypt(self, pub: paillier.PaillierPublicKey):
		for i in range(len(self.data)):
			self.data[i] = pub.encrypt(self.data[i], 1e-10)
		self.is_enc = True

	def decrypt(self, dec: paillier.PaillierPrivateKey):
		for i in range(len(self.data)):
			self.data[i] = dec.decrypt(self.data[i])
		self.is_enc = False

	def get_cosine_similarity(self, emb):
		if emb.is_enc and self.is_enc:
			raise Exception("method must have not encrypted embedding")
		if len(emb.data) != len(self.data):
			raise Exception("lengths of embeddings must be equals")
		cosine_sim = self.data[0] * emb.data[0]
		for i in range(1, len(self.data)):
			cosine_sim += self.data[i] * emb.data[i]
		return cosine_sim

	def json(self):
		if self.is_enc:
			l = []
			for i in range(len(self.data)):
				l.append((self.data[i].ciphertext(False), self.data[i].exponent))
			return {"data": l}
		else:
			return {"data": list(self.data)} 

class EmbeddingManager:
	def __init__(self, src_path: str, emb_path: str):
		self.src_dir = src_path
		self.emb_dir = emb_path
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.mtcnn = MTCNN(
					image_size=160, margin=0, min_face_size=20,
					thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
					device=self.device
			)
		self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
	
	def get_emb(self, img_path: str):
		with Image.open(img_path) as img:
			cropped, prob = self.mtcnn(img, return_prob = True)
		if cropped == None or prob < 0.5:
			print(f"image {file} don't has face")
		emb = self.resnet(cropped.unsqueeze(0))
		return Embedding(emb[0].tolist())

	def update_src_emb(self, pub_key: paillier.PaillierPublicKey):
		# delete old embiddings
		for file in Path(self.emb_dir).iterdir():
			file.unlink()
		# create new embiddings
		for file in Path(self.src_dir).iterdir():
			with Image.open(file) as img:
				cropped, prob = self.mtcnn(img, return_prob = True)
				if cropped == None or prob < 0.5:
					print(f"image {file} don't has face")
				emb = self.resnet(cropped.unsqueeze(0))
				e = Embedding(emb[0].tolist())
				e.encrypt(pub_key)
				with open(self.emb_dir + "/" + file.name + ".json", "w") as j:
					json.dump(e.json(), j)

	def get_embs(self, pub_key: paillier.PaillierPublicKey):
		embs = []
		for file in Path(self.emb_dir).iterdir():
			with open(file, "r") as jd:
				j = json.load(jd)
				embs.append(Embedding(j["data"], True, {"pub": pub_key}))
		return embs



if __name__ == "__main__":
	if len(argv) < 3:
		print(f"Usage: {argv[0]} KEYS_PATH PHOTO [update]")
		exit(0)
	print("Start scan")

	pub_file = argv[1] + "/pub.key"
	secret_file = argv[1] + "/secret.key"
	trg_file = argv[2]
	update_bool = False
	if len(argv) > 3 and argv[3] == "update":
		update_bool = True

	pub_key = None
	secret_key = None

	with open(pub_file, "r") as file:
		pub_j = json.load(file)
		if "n" not in pub_j.keys():
			raise Exception("bad input public key")
		pub_key = paillier.PaillierPublicKey(pub_j["n"])

	with open(secret_file, "r") as file:
		sec_j = json.load(file)
		if "p" not in sec_j.keys() or "q" not in sec_j.keys():
			raise Exception("bad input secret key")
		secret_key = paillier.PaillierPrivateKey(pub_key, sec_j["p"], sec_j["q"])

	emb_mgr = EmbeddingManager("src_faces", "emb_faces")

	if len(argv) > 3 and update_bool:
		emb_mgr.update_src_emb(pub_key)

	src_embs = emb_mgr.get_embs(pub_key)

	if len(src_embs) == 0:
		print("Force update source embedding")
		emb_mgr.update_src_emb(pub_key)
		src_embs = emb_mgr.get_embs(pub_key)

	print(f"Has {len(src_embs)} peoples in database")

	trg_emb = emb_mgr.get_emb(trg_file)

	for i in range(len(src_embs)):
		print(secret_key.decrypt(src_embs[i].get_cosine_similarity(trg_emb)))

	# data = [1, 2, 3, 4, 5]
	# # data_tmp = [1, 2, 3, 4, 5]

	# emb1 = Embedding(data, False)
	# # emb1.normalize()
	# emb1.encrypt(pub_key)

	# # emb2 = Embedding(data_tmp)
	# # emb2.normalize()
	
	# # print(repr(emb1.data[0]))



	# # data  = pub_key.encrypt(1)
	# data_j = {"data": convert_emb_to_list(emb1)}
	# with open("emb.json", "w") as file:
	# 	json.dump(data_j, file)
	# with open("emb.json", "r") as file:
	# 	data_j = json.load(file)
	# emb2 = Embedding(data_j["data"], True, {"pub": pub_key})
	# emb2.decrypt(secret_key)
	# print(emb2.data)
	# # print(secret_key.decrypt(emb1.get_cosine_similarity(emb2)))
