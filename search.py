import duckdb
from sentence_transformers import SentenceTransformer

# 1. Ngarkimi i modelit (Duhet të jetë i njëjti me atë të insertimit)
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
print("Duke ngarkuar modelin...")
model = SentenceTransformer(MODEL_NAME)

# 2. Lidhja me databazën DuckDB
con = duckdb.connect('albanian_vectors.db')

# Ngarkojmë extension-in për kërkim vektorial
con.execute("LOAD vss;")

def shto_kerkim(pyetja):
    print(f"\nDuke kërkuar për: '{pyetja}'")
    
    # Kthejmë fjalën/fjalinë e kërkuar në vektor (embedding)
    query_vec = model.encode([pyetja])[0].tolist()
    
    # 3. Kërkimi në DuckDB duke përdorur Cosine Similarity
    # FLOAT[384] duhet të korrespondojë me dimensionet e modelit MiniLM
    rezultatet = con.execute("""
        SELECT cleaned_text, array_cosine_similarity(embedding, ?::FLOAT[384]) as score
        FROM dictionary_embeddings
        ORDER BY score DESC
        LIMIT 5
    """, [query_vec]).fetchall()
    
    print("-" * 30)
    for i, (teksti, skori) in enumerate(rezultatet):
        print(f"{i+1}. [Ngjashmëria: {skori:.4f}]")
        print(f"Teksti: {teksti[:200]}...\n")

# Ekzekutimi i kërkimit
if __name__ == "__main__":
    search_query = input("Shkruani fjalën që dëshironi të kërkoni në fjalor: ")
    shto_kerkim(search_query)
    con.close()