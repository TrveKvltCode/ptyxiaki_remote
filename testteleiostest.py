from toolkit import embed_pdf, retrieve_documents


PDF_DIR = r"C:\Users\mpamp\PycharmProjects\ptyxiaki\Study Guide 2024.pdf"
CHROMADB_DIR = r"C:\Users\mpamp\PycharmProjects\ptyxiaki\testembeddingsdir"

#embed_pdf(pdf_path=PDF_DIR, vectorstore_dir=CHROMADB_DIR)


docs = retrieve_documents(query="Η εκπαιδευτική διδασκαλία κάθε μαθήματος περιλαμβάνει μια ή περισσότερες από τις παρακάτω μορφές", vectorstore_dir=CHROMADB_DIR)

for doc in docs:
    print(type(doc))
    print(doc.metadata)
    print(doc.page_content)