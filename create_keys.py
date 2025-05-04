from phe import paillier
from pathlib import Path
import json

pub_key,priv_key = paillier.generate_paillier_keypair()

pub = {"g": pub_key.g, "n": pub_key.n}
priv = {"p": priv_key.p, "q": priv_key.q}

if not Path("./keys").exists():
	Path("./keys").mkdir(parents=True, exist_ok=True)

with open("./keys/pub.key", "w") as file:
	json.dump(pub, file)

with open("./keys/secret.key", "w") as file:
	json.dump(priv, file)