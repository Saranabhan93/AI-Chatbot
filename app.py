import os
import streamlit as st
from dotenv import load_dotenv
import pdfplumber

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------------------------
# 1. إعداد المفاتيح والصفحة
# -------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG AI Assistant", page_icon="📚")
st.title("📚 AI Assistant with RAG (PDF Q&A)")

if not GOOGLE_API_KEY:
    st.warning("الرجاء التأكد من إضافة مفتاح GOOGLE_API_KEY في ملف .env")
    st.stop()

# -------------------------------------------------
# 2. رفع ملف PDF
# -------------------------------------------------
uploaded_file = st.file_uploader("اختر ملف PDF", type=["pdf"])

if uploaded_file is not None:

    # حفظ الملف مؤقتاً
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("تم رفع الملف بنجاح! جاري المعالجة...")

    # -------------------------------------------------
    # 3. استخراج النص من الـ PDF
    # -------------------------------------------------
    documents = []
    with pdfplumber.open(temp_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"page": i}))

    if not documents:
        st.error("ما قدرت أستخرج أي نص من الملف. جربي ملف PDF تاني.")
        st.stop()

    # -------------------------------------------------
    # 4. تقسيم النص إلى مقاطع (chunks)
    # -------------------------------------------------
    chunk_size = 1000
    chunks = []
    for doc in documents:
        text = doc.page_content
        for i in range(0, len(text), chunk_size):
            piece = text[i:i + chunk_size]
            chunks.append(Document(page_content=piece, metadata=doc.metadata))

    # -------------------------------------------------
    # 5. توليد الـ embeddings محلياً (بدون Google)
    #    هيك بنتجنب مشكلة ترميز النصوص العربية
    # -------------------------------------------------
    with st.spinner("جاري تجهيز المستند..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # -------------------------------------------------
    # 6. نموذج الإجابة (Gemini)
    # -------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    # -------------------------------------------------
    # 7. سؤال وجواب
    # -------------------------------------------------
    user_query = st.text_input("اسأل أي سؤال حول محتوى ملف الـ PDF:")

    if user_query:
        with st.spinner("جاري البحث في المستند وصياغة الإجابة..."):
            relevant_docs = retriever.invoke(user_query)
            context = "\n".join(doc.page_content for doc in relevant_docs)

            prompt = (
                f"استناداً إلى المعلومات التالية فقط:\n{context}\n\n"
                f"أجب عن السؤال التالي: {user_query}"
            )
            response = llm.invoke(prompt)

            # الموديلات الجديدة بترجع المحتوى كقائمة أجزاء، فبنستخرج النص بس منها
            if isinstance(response.content, str):
                answer_text = response.content
            else:
                answer_text = "\n".join(
                    part.get("text", "")
                    for part in response.content
                    if isinstance(part, dict) and part.get("type") == "text"
                )

        st.markdown("### الإجابة:")
        st.write(answer_text)

    # تنظيف الملف المؤقت
    os.remove(temp_path)