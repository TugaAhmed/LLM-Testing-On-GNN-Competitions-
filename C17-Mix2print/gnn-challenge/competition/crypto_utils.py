import os
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

def generate_key_pair(private_path="private_key.pem", public_path="public_key.pem"):
    """
    Generates a new RSA private/public key pair.
    Saves private key to `private_path` (KEEP SECRET!)
    Saves public key to `public_path` (DISTRIBUTE!)
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Save Private Key
    with open(private_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # Save Public Key
    public_key = private_key.public_key()
    with open(public_path, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    
    print(f"Generated keys:\n  Private: {private_path}\n  Public:  {public_path}")

def encrypt_file(file_path, public_key_path, team_name):
    """
    Encrypts a file for submission.
    
    Format of output file (binary):
    [256 bytes: Encrypted Fernet Key] + [N bytes: Encrypted Data]
    
    The Data itself contains: "TEAM_NAME\nCSV_CONTENT"
    This binds the submission to the team name to prevent theft.
    """
    # 1. Load Public Key
    with open(public_key_path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read())

    # 2. Generate a random Fernet key (symmetric)
    fernet_key = Fernet.generate_key()
    f = Fernet(fernet_key)

    # 3. Read file content and prepend metadata
    with open(file_path, "rb") as file:
        file_data = file.read()
    
    # Bind team name to data
    payload = f"{team_name}\n".encode('utf-8') + file_data
    encrypted_data = f.encrypt(payload)

    # 4. Encrypt the Fernet key with the Public Key (RSA)
    encrypted_key = public_key.encrypt(
        fernet_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # 5. Return combined package
    return encrypted_key + encrypted_data

def decrypt_submission(encrypted_path, private_key_pem_bytes):
    """
    Decrypts a submission file.
    Returns: (team_name, csv_content_bytes)
    """
    # 1. Load Private Key from bytes (e.g., from env var)
    private_key = serialization.load_pem_private_key(
        private_key_pem_bytes,
        password=None
    )

    # 2. Read Encrypted File
    with open(encrypted_path, "rb") as f:
        data = f.read()

    # Split: Key (256 bytes) | Data (Rest)
    if len(data) < 256:
        raise ValueError("File too small/corrupted.")
        
    encrypted_key = data[:256]
    encrypted_content = data[256:]

    # 3. Decrypt the Fernet Key
    fernet_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

    # 4. Decrypt the Content
    f = Fernet(fernet_key)
    decrypted_payload = f.decrypt(encrypted_content)

    # 5. Separate Team Name from CSV
    try:
        split_idx = decrypted_payload.index(b'\n')
        team_name = decrypted_payload[:split_idx].decode('utf-8')
        csv_content = decrypted_payload[split_idx+1:]
        return team_name, csv_content
    except ValueError:
        raise ValueError("Decrypted payload format invalid (missing team header).")
