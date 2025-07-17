"""
Vector Database Builder
FAİSS: is a library for efficient similarity search and clustering of dense vectors.
"""

# import libraries
import os
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer #embeding
import faiss # vektör veri tabanı
import numpy as np
import pickle # veri tabanı dosyalarını kaydetmek için


# program için dosya olarak .pdf yükliyelim
# .pdf den metiin donusumu yapmamız gerek
def extract_text_from_pdf(pdf_path):
    """
         pdf dosyasından metin çıkarma
    """
    doc = fitz.open(pdf_path)
    text= ""
    for page in doc:
        text += page.get_text()

    return text

#print(extract_text_from_pdf(".\data\Consulting_Agreement.pdf"))

#uzun metni daha küçük parçalara ayırma
def chunks_text(text, max_legth=500):
    """
        metni belirtilen karakter uzunluğuna göre böl
    """
    chunks = []
    current = ""
    for line in text.split('\n'):
        if len(current) + len(line) < max_legth:
            current += " " + line.strip() 
        else:
            chunks.append(current.strip())
            current = line.strip() 
    if current:
        chunks.append(current.strip())

    return chunks
    
text_dummy = extract_text_from_pdf(".\data\Consulting_Agreement.pdf")
print(chunks_text(text_dummy, max_legth=500))

#sentence transformers ile embeding
#https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
model = SentenceTransformer("all-MiniLM-L6-v2")

#pdf yolunu belirt
pdf_file_path = ".\data\Consulting_Agreement.pdf"

#pdf den metin çıkartalim
text = extract_text_from_pdf(pdf_file_path)

# metni chunk lara bölelim 
chunks = chunks_text(text, max_legth=500)

#her chunk için embedding (vektorel temsil) oluşturalım
embeddings = model.encode(chunks)

print(f"embeddings shape: {embeddings.shape}")  # (n_chunks, embedding_dim)

#faiis index oluştur 
dimension = embeddings.shape[1]  # embedding ( vektor ) boyutu
index = faiss.IndexFlatL2(dimension)  # L2 mesafesi kullanarak benzerlik arama
index.add(np.array(embeddings))  # embeddingleri indekse ekle

#faiss indexi ve chunkları kaydet
faiss.write_index(index, "data/contract_index.faiss")
with open("data/contract_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("faiss index ve chunklar kaydedildi.")