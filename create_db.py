"""
create_db.py
------------
Run this ONCE to set up the MySQL database for LexiSum AI.
Usage: python create_db.py

Requirements: pip install mysql-connector-python
"""

import mysql.connector

# --- CHANGE THESE TO MATCH YOUR LOCAL MYSQL SETUP ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "charset": "utf8"
}

DB_NAME = "lexisum_db"

# All SQL statements to build the schema
SCHEMA_SQL = [
    # 1. Users table
    """
    CREATE TABLE IF NOT EXISTS users (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        full_name   VARCHAR(100)        NOT NULL,
        email       VARCHAR(150)        NOT NULL UNIQUE,
        password    VARCHAR(255)        NOT NULL,
        created_at  TIMESTAMP           DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
    """,

    # 2. Admins table
    """
    CREATE TABLE IF NOT EXISTS admins (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        username    VARCHAR(50)         NOT NULL UNIQUE,
        password    VARCHAR(255)        NOT NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
    """,

    # 3. Documents table — stores upload metadata and results
    """
    CREATE TABLE IF NOT EXISTS documents (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        user_id         INT             NOT NULL,
        filename        VARCHAR(255)    NOT NULL,
        status          ENUM('processing', 'completed', 'error') DEFAULT 'processing',
        english_summary TEXT,
        tamil_summary   TEXT,
        uploaded_at     TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
    """,

    # 4. Seed a default admin account (username: admin / password: admin)
    """
    INSERT IGNORE INTO admins (username, password)
    VALUES ('admin', 'admin123');
    """,

    # 5. Seed a test user account (email: jai@gmail.com / password: 1234)
    """
    INSERT IGNORE INTO users (full_name, email, password)
    VALUES ('Jai', 'jai@gmail.com', '1234');
    """,
]


def main():
    print("Connecting to MySQL...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Create the database with strict utf8 charset
    print(f"Creating database '{DB_NAME}' (utf8)...")
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
        f"CHARACTER SET utf8 COLLATE utf8_general_ci;"
    )
    cursor.execute(f"USE {DB_NAME};")

    # Run each table/seed statement
    for sql in SCHEMA_SQL:
        cursor.execute(sql)
        conn.commit()

    print("\n✅ Database setup complete!")
    print(f"   Database : {DB_NAME}")
    print(f"   Tables   : users, admins, documents")
    print(f"   Defaults : admin/admin  |  jai@gmail.com/1234")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
