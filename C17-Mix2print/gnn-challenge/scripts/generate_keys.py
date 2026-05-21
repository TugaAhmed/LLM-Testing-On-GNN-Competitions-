from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

def generate_keys():
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Save private key (SECRET! Put in GitHub Secrets)
    with open("submission_private.pem", "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Save public key (PUBLIC! Put in repo)
    public_key = private_key.public_key()
    with open("data/public/submission.key", "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    print("[OK] Keys Generated!")
    print("---------------------------------------------------")
    print("1. [PRIVATE] submission_private.pem -> Copy content to GitHub Secret 'SUBMISSION_PRIVATE_KEY'")
    print("2. [PUBLIC]  data/public/submission.key -> Commit this file to the repo")
    print("---------------------------------------------------")

if __name__ == "__main__":
    generate_keys()
