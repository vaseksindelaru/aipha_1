# inspect_chroma.py
import chromadb

# Conectar a la base de datos
client = chromadb.PersistentClient(path="./shadow_storage/vector_db")

# Listar todas las colecciones
collections = client.list_collections()
print("Colecciones disponibles:")
for coll in collections:
    print(f" - {coll.name}")

# Si quieres ver el contenido de una colección específica
collection = client.get_collection("aipha_shadow_collection")

# Contar documentos
count = collection.count()
print(f"\nTotal de documentos: {count}")

# Ver los primeros 5 documentos
if count > 0:
    results = collection.peek(limit=5)
    print("\nPrimeros 5 documentos:")
    for i in range(len(results['ids'])):
        doc_id = results['ids'][i]
        metadata = results['metadatas'][i]
        print(f"Documento {i+1}:")
        print(f"  ID: {doc_id}")
        print(f"  Metadata: {metadata}")
        print(f"  Contenido: {results['documents'][i][:100]}...")  # Primeros 100 caracteres
        print("---")