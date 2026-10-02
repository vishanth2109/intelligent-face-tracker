import sqlite3

connection = sqlite3.connect("database/visitors.db")
cursor = connection.cursor()

entry_count = cursor.execute(
    "SELECT COUNT(*) FROM events WHERE event_type = ?",
    ("ENTRY",)
).fetchone()[0]

exit_count = cursor.execute(
    "SELECT COUNT(*) FROM events WHERE event_type = ?",
    ("EXIT",)
).fetchone()[0]

unique_visitors = cursor.execute(
    "SELECT COUNT(DISTINCT face_id) FROM persons"
).fetchone()[0]

print("================================")
print("DATABASE VALIDATION")
print("================================")
print(f"ENTRY events      : {entry_count}")
print(f"EXIT events       : {exit_count}")
print(f"Unique visitors   : {unique_visitors}")
print("================================")

connection.close()