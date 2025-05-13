from types import NoneType
from facenet_pytorch import MTCNN, InceptionResnetV1
from phe import paillier
from sys import argv
from math import sqrt
from pathlib import Path
from progress.bar import IncrementalBar
import torch
from PIL import Image
import json
import typing
import os
import sys
import time
import cv2

from torch.nn.modules import EmbeddingBag

def write_int(i: int, s: int,  file):
	if i < 0:
		file.write(int(1).to_bytes(1, byteorder="big"))
	else:
		file.write(int(0).to_bytes(1, byteorder="big"))
	file.write(abs(i).to_bytes(s, byteorder="big"))

def read_int(s: int, file):
	sig = int.from_bytes(file.read(1), byteorder="big")
	abs_v = int.from_bytes(file.read(s), byteorder="big")
	if bool(sig):
		return -1 * abs_v
	else:
		return abs_v


class Embedding:
	def __init__(self, data: list, is_enc: bool = False, pub_key: paillier.PaillierPublicKey = None):
		self.is_enc = is_enc
		if not is_enc:
			if len(data) == 0:
				raise Exception("data must have at least 1 element")
			self.data = data
		else:
			self.data = []
			for i in range(len(data)):
				self.data.append(paillier.EncryptedNumber(pub_key, data[i][0], data[i][1]))

	def normalize(self):
		l = 0
		for val in self.data:
			l += val ** 2
		l = sqrt(l)
		for i in range(len(self.data)):
			self.data[i] = self.data[i] / l

	def encrypt(self, pub: paillier.PaillierPublicKey):
		bar = IncrementalBar('encrypting', max = len(self.data), suffix="%(percent)d%% %(elapsed_td)s")
		for i in range(len(self.data)):
			self.data[i] = pub.encrypt(self.data[i], 1e-10)
			bar.next()
		print()
		self.is_enc = True

	def decrypt(self, dec: paillier.PaillierPrivateKey):
		bar = IncrementalBar('decrypting', max = len(self.data), suffix="%(percent)d%%")
		for i in range(len(self.data)):
			self.data[i] = dec.decrypt(self.data[i])
			bar.next()
		print()
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

	staticmethod
	def _save_value(value, file):
		size = sys.getsizeof(value)
		write_int(size, 4, file)
		write_int(value, size, file)

	staticmethod
	def _load_value(file) -> int:
		s = read_int(4, file)
		assert s > 0
		return read_int(s, file)

	staticmethod
	def save(emb, path):
		with open(path, "wb") as file:
			write_int(int(emb.is_enc), 1, file)
			l = len(emb.data)
			write_int(l, 4, file)
			for i in range(len(emb.data)):
				if emb.is_enc:
					Embedding._save_value(emb.data[i].ciphertext(False), file)
					Embedding._save_value(emb.data[i].exponent, file)
				else:
					Embedding._save_value(emb.data[i], file)

	staticmethod
	def load(path, pub_key: paillier.PaillierPublicKey):
		with open(path, "rb") as file:
			is_enc = bool(read_int(1, file))
			count = read_int(4, file)
			data = []
			for i in range(count):
				if not is_enc:
					data.append(Embedding._load_value(file))
				else:
					c = Embedding._load_value(file)
					e = Embedding._load_value(file)
					data.append((c, e))
			if not is_enc:
				return Embedding(data, False)
			else:
				return Embedding(data, True, pub_key)

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
			print(f"Scanning face from {file}")
			e = self.get_emb(file)
			print(f"Encryption face from {file}")
			e.encrypt(pub_key)
			path = self.emb_dir + "/" + file.name + ".bin"
			print(f"Saving face from {file}")
			Embedding.save(e, path)

	def get_embs(self, pub_key: paillier.PaillierPublicKey):
		embs = []
		for file in Path(self.emb_dir).iterdir():
			embs.append((Embedding.load(file, pub_key), file))
		return embs

class Client:
	def __init__(self):
		self.cam = cv2.VideoCapture(0)

	def who_are_you(self) -> str:
		return input("Your name: ")

	def get_sec_similary(self, sec_src_emb, emb_mgr):
		print("Cheese!")
		time.sleep(0.5)
		ret, frame = self.cam.read()
		print("Processing...")
		cv2.imwrite("photo.png", frame)
		trg_emb = emb_mgr.get_emb("photo.png")
		return sec_src_emb.get_cosine_similarity(trg_emb)

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

	if not Path("emb_faces").exists():
		Path("./emb_faces").mkdir(parents=True, exist_ok=True)

	if not Path("src_faces").exists():
		Path("./src_faces").mkdir(parents=True, exist_ok=True)

	emb_mgr = EmbeddingManager("src_faces", "emb_faces")

	if len(argv) > 3 and update_bool:
		emb_mgr.update_src_emb(pub_key)

	src_embs = emb_mgr.get_embs(pub_key)

	if len(src_embs) == 0:
		print("Force update source embedding")
		emb_mgr.update_src_emb(pub_key)
		src_embs = emb_mgr.get_embs(pub_key)

	print(f"Has {len(src_embs)} peoples in database")

	if trg_file == "camera":
		try:
			client = Client()
			while True:
				name = client.who_are_you()
				src_emb = None
				for emb in src_embs:
					if name in str(emb[1]):
						src_emb = emb
						break
				if not isinstance(src_emb, NoneType):
					t = time.process_time()
					print(f"Similary with {src_emb[1]}: {abs(secret_key.decrypt(client.get_sec_similary(src_emb[0], emb_mgr)))}")
					print(f"Time: {time.process_time() - t}s")
				else:
					print("Unknown user")
		except KeyboardInterrupt:
			exit(0)
	else:
		print("Getting face embidding")
		trg_emb = emb_mgr.get_emb(trg_file)

		for i in range(len(src_embs)):
			t = time.process_time()
			print(f"Similary with {src_embs[i][1]}: {abs(secret_key.decrypt(src_embs[i][0].get_cosine_similarity(trg_emb)))}")
			print(f"Time: {time.process_time() - t}s")
