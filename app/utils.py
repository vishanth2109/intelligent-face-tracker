import numpy as np


def cosine_similarity(embedding1, embedding2):

    embedding1 = np.asarray(
        embedding1,
        dtype=np.float32
    )

    embedding2 = np.asarray(
        embedding2,
        dtype=np.float32
    )

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    embedding1 = embedding1 / norm1
    embedding2 = embedding2 / norm2

    return float(
        np.dot(embedding1, embedding2)
    )


def find_best_match(
    query_embedding,
    stored_faces,
    threshold=0.45
):

    if query_embedding is None:
        return None, -1.0

    best_face_id = None
    best_similarity = -1.0

    for face in stored_faces:

        stored_embedding = face.get("embedding")

        if stored_embedding is None:
            continue

        similarity = cosine_similarity(
            query_embedding,
            stored_embedding
        )

        print(
            f"  Comparing with {face['face_id']}"
            f" -> {similarity:.4f}"
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_face_id = face["face_id"]

    if best_similarity >= threshold:

        return (
            best_face_id,
            best_similarity
        )

    return None, best_similarity

