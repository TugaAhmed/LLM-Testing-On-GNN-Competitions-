import os
import sys
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet

def decrypt_workflow(enc_path, private_key_pem):
    """
    Workflow helper to decrypt submission.
    """
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None
    )

    with open(enc_path, "rb") as f:
        data = f.read()

    # Split: Key (256 bytes) | Data (Rest)
    encrypted_key = data[:256]
    encrypted_content = data[256:]

    # Decrypt AES Key
    try:
        fernet_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    except Exception as e:
        print("❌ Decryption Failed: Invalid RSA Key (or corrupted file).")
        sys.exit(1)

    # Decrypt Content
    f = Fernet(fernet_key)
    try:
        decrypted_payload = f.decrypt(encrypted_content)
    except Exception as e:
        print("❌ Decryption Failed: Invalid AES Data.")
        sys.exit(1)

    # Split Team Name
    try:
        split_idx = decrypted_payload.index(b'\n')
        team_name = decrypted_payload[:split_idx].decode('utf-8')
        csv_content = decrypted_payload[split_idx+1:]
    except ValueError:
        print("❌ Invalid Payload: Missing Team Name header.")
        sys.exit(1)

    # Verify Team Name (Optional additional check, but good practice)
    print(f"[OK] Decrypted submission for Team: {team_name}")
    
    # Save CSV
    with open("predictions.csv", "wb") as f:
        f.write(csv_content)
    
    print("[OK] Saved to predictions.csv")
    return team_name

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decrypt_workflow.py submission.enc")
        sys.exit(1)
    
    enc_path = sys.argv[1]
    pem_str = os.environ.get("SUBMISSION_PRIVATE_KEY")
    
    if not pem_str:
        print("❌ Error: SUBMISSION_PRIVATE_KEY environment variable not set.")
        sys.exit(1)
        
    decrypt_workflow(enc_path, pem_str)
