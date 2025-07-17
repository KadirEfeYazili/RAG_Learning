"""
problem tanımı : Sözleşme asistanı
                 Kullanıcının yükeldiği bir sözeleşme dosyasından içerik çıkarmak
                 bu içeriği vektörel olarak temsil edelim yani (embedding yapalım)
                 faiss kullanarak hızlı arama yapabilen bir vektör veri tabanı oluştur
                 kullanıcı sorularını al, sonra git db den bilgiyi getir, sonra gpt 3.5 ile cevapla

kullanılan teknojiler:
    - embedding: metni vektörleştirme
    - faiss: hizli beznerlik aramasi icin vektor veri tabanı
    - gpt 3.5: metin üretimi ve cevaplama

RAG: Retrieval Augmented Generation : dil modellerine bilgi desteği sağlayan bir teknik
    - kullanici sorularini al, ilgili bilgileri veri tabanından getir,sonra gpt ile cevapla
    - retrieval: kullanici sorusu embeddinge dönüştürülür, faiss (db) üzreinden en alaklıiçerik (chunk) getiriliyor
    - augmentation: zenginleştirme, bulunan metin parcalari llm'in anlyabileceği bir formata dönüştürüyor
    - generation: dil modeli bu bilgilerle matnıklı yanıt üretir
          - 1) tarih
          - 2) ücret
          - 3) taraflar TechNova Solutions Ltd. - John A. Carter
    
Neden RAG kullanıyoruz?
   - modelin eğitim verisinde olmayan özel belgelerle çalışmasını sağlamak için
   - belgeler güncellendiğinde tekrar model eğitimi gerekmediği için

Plan/program
     - sozlesme belgesinin hazirlanmasi
     - metin cikarma ve parcalama
     - embedding ve faiss ile vector db oluşturma
     - soru cevap sistemi 

instal libraries: freeze
"""
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()

# Google Gemini API anahtarını al
api_key = os.getenv("GEMINI_API_KEY")

# Google Gemini yapılandırması
genai.configure(api_key=api_key)

# Gemini modelini başlat
model = genai.GenerativeModel("gemini-1.5-flash")

# Embedding modeli (SentenceTransformer)
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index dosyasını yükle
index = faiss.read_index("./data/contract_index.faiss")

# Chunklanmış metinleri yükle
with open("./data/contract_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Soru-cevap döngüsü
while True:
    print("\nÇıkmak için 'exit' yazabilirsiniz.")
    question = input("\nSorunuzu Giriniz (ENG): ")
    
    if question.lower() == "exit":
        print("Çıkılıyor...")
        break

    # Soruyu vektöre dönüştür
    question_embedding = model_embed.encode([question])

    # FAISS ile en yakın 3 chunk'ı bul
    k = 3
    distances, indices = index.search(np.array(question_embedding), k)

    # Chunk'ları al ve birleştir
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n------\n".join(retrieved_chunks)

    prompt = f"""You are a contract assistant. Based on the contract context below, 
            answer the user's question clearly.

            Context: {context}

            Question: {question}

            Answer:
            """



    # Gemini modeliyle yanıt oluştur
    response = model.generate_content(prompt)

    # Yanıtı yazdır
    print("\nAI Asistan Cevap:\n", response.text.strip())