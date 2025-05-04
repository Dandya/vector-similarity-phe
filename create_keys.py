from phe import paillier
import json

pub_key,priv_key = paillier.generate_paillier_keypair()

pub = {"g": pub_key.g, "n": pub_key.n}
priv = {"p": priv_key.p, "q": priv_key.q}

with open("pub.key", "w") as file:
	json.dump(pub, file)

with open("secret.key", "w") as file:
	json.dump(priv, file)