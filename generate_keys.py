# generate_keys.py

import bcrypt
import sys

# Check if a password was provided as an argument
if len(sys.argv) < 2:
    print("Usage: python generate_keys.py \"<your-password>\"")
    sys.exit(1)

# Get the password from the command-line argument
password = sys.argv[1].encode('utf-8') # bcrypt requires bytes

# Generate the hash
hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())

# Print the final hashed password as a string
print(hashed_password.decode())