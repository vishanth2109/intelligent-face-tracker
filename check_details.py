import sqlite3

connection = sqlite3.connect("database/visitors.db")
cursor = connection.cursor()

print("\n================================")
print("PERSONS")
print("================================")

persons = cursor.execute("""
    SELECT face_id, first_seen, last_seen
    FROM persons
    ORDER BY id
""").fetchall()

for person in persons:
    print(person)


print("\n================================")
print("EVENTS")
print("================================")

events = cursor.execute("""
    SELECT face_id, track_id, event_type, timestamp
    FROM events
    ORDER BY id
""").fetchall()

for event in events:
    print(event)


print("\n================================")
print("EVENT COUNT PER FACE")
print("================================")

counts = cursor.execute("""
    SELECT
        face_id,
        SUM(CASE WHEN event_type = 'ENTRY' THEN 1 ELSE 0 END),
        SUM(CASE WHEN event_type = 'EXIT' THEN 1 ELSE 0 END)
    FROM events
    GROUP BY face_id
    ORDER BY face_id
""").fetchall()

for row in counts:
    print(
        f"{row[0]} -> "
        f"ENTRY={row[1]}, EXIT={row[2]}"
    )


connection.close()