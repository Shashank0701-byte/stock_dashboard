import sqlite3

# Connect to the database (this will create the file if it doesn't exist)
conn = sqlite3.connect('watchlist.db')
cursor = conn.cursor()

# Create the watchlists table
# This table will store the username and their list of tickers
cursor.execute('''
    CREATE TABLE IF NOT EXISTS watchlists (
        username TEXT PRIMARY KEY,
        tickers TEXT NOT NULL
    )
''')

print("Database and 'watchlists' table created successfully.")

# Save the changes and close the connection
conn.commit()
conn.close()