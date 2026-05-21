import sys
import os
import argparse
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet
import base64

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

def load_public_key(key_path):
    with open(key_path, "rb") as key_file:
        return serialization.load_pem_public_key(key_file.read())

def encrypt_file(file_path, public_key_path, team_name, output_path):
    """
    Encrypts a file for submission using Hybrid Encryption.
    1. Generate AES (Fernet) key.
    2. Encrypt file with Fernet.
    3. Encrypt Fernet key with RSA Public Key.
    4. Bundle: [RSA-Encrypted-Key (256 bytes)] + [AES-Encrypted-Data]
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
        
    print(f"[INFO] Encrypting '{file_path}' for Team '{team_name}'...")

    # Load Public Key
    public_key = load_public_key(public_key_path)

    # Generate AES Key
    fernet_key = Fernet.generate_key()
    fernet = Fernet(fernet_key)

    # Read CSV
    with open(file_path, "rb") as f:
        file_data = f.read()

    # Prepend Team Name (binds submission to identity)
    # Format: "TeamName\nCSV_Data"
    payload = f"{team_name}\n".encode('utf-8') + file_data
    
    # AES Encrypt Payload
    encrypted_data = fernet.encrypt(payload)

    # RSA Encrypt AES Key
    encrypted_key = public_key.encrypt(
        fernet_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # Combine
    final_blob = encrypted_key + encrypted_data
    
    # Save
    with open(output_path, "wb") as f:
        f.write(final_blob)

    print(f"[OK] Success! Generated encrypted submission: {output_path}")
    print("[INFO] Upload ONLY this .enc file to your Pull Request.")

def main():
    parser = argparse.ArgumentParser(description="Encrypt your submission file for the GNN Competition.")
    parser.add_argument("file", help="Path to predictions.csv")
    parser.add_argument("--team", required=True, help="Your Team Name (must match your folder name)")
    parser.add_argument("--public-key", default="data/public/submission.key", help="Path to public key")
    parser.add_argument("--output", default="submission.enc", help="Output filename")

    args = parser.parse_args()
    
    # Path correction relative to repo root if run from elsewhere
    key_path = os.path.join(ROOT_DIR, args.public_key)
    
    if not os.path.exists(key_path):
        print(f"❌ Error: Public key not found at {key_path}. Are you in the repo root?")
        sys.exit(1)

    encrypt_file(args.file, key_path, args.team, args.output)

if __name__ == "__main__":
    main()
