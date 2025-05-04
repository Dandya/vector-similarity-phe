from phe import paillier
from sys import argv
from math import sqrt
import json
import typing

class Embedding:
	def __init__(self, data):
		if len(data) == 0:
			raise Exception("data must have at least 1 element")
		self.data = data
		self.is_enc = False

	def normalize(self):
		l = 0
		for val in self.data:
			l += val ** 2
		l = sqrt(l)
		for i in range(len(self.data)):
			self.data[i] = self.data[i] / l

	def encrypt(self, pub: paillier.PaillierPublicKey):
		for i in range(len(self.data)):
			self.data[i] = pub.encrypt(self.data[i])
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

if __name__ == "__main__":
	if len(argv) < 3:
		print(f"Usage: {argv[0]} PUB_KEY SECRET_KEY")
		exit(0)

	pub_file = argv[1]
	secret_file = argv[2]

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

	data = [1, 2, 3, 4, 5]
	data_tmp = [1, 2, 3, 4, 5]

	emb1 = Embedding(data)
	emb1.normalize()
	emb1.encrypt(pub_key)

	emb2 = Embedding(data_tmp)
	emb2.normalize()
	print(secret_key.decrypt(emb1.get_cosine_similarity(emb2)))
