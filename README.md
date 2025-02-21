# 📌 IAB Taxonomy Classification Using RAG

## 📖 Overview  
This project explores the use of a **hierarchical knowledge graph-based Retrieval-Augmented Generation (RAG) model** for **webpage classification** into the **IAB Content Taxonomy**. By leveraging **Neo4j for graph representation**, **context-constrained synonyms**, **multilingual embeddings**, and **LLM-driven summarization**, we enhance classification accuracy while reducing **semantic ambiguity and misclassification**.  

## 🚀 Key Features  
✔️ **Graph-Based Retrieval** – Uses **Neo4j** to structure hierarchical taxonomy relationships.  
✔️ **Context-Aware Synonyms** – LLM-driven **synonym generation** for enhanced precision.  
✔️ **Multi-Language Support** – Processes data in **English, German, French, and Chinese**.  
✔️ **Weight-Based Node Refinement** – Ensures **cluster loyalty and reduced ambiguity**.  
✔️ **LLM Summarization Enhancement** – Combines **original and summarized text** for better classification.  
✔️ **Scalable & Optimized** – Utilizes **vector search, embeddings, and aggregation-based ranking**.  

---

## 📊 Project Objectives  
🔹 **Improve IAB Taxonomy classification** by integrating **graph-based retrieval** and **semantic embeddings**.  
🔹 **Enhance retrieval quality** through **context-aware synonyms** and **LLM-driven refinements**.  
🔹 **Address semantic overlap & feature ambiguity** using **weight-based node scoring**.  
🔹 **Evaluate classification accuracy** across **multiple NLP models** and different weight configurations.  

---

## 🛠️ Tech Stack  
🔹 **Languages & Frameworks:** Python, Neo4j, PyTorch, Transformers, TensorFlow  
🔹 **Database:** Neo4j Graph Database, PostgreSQL  
🔹 **Machine Learning:** LLMs (GPT, BERT, LaBSE, mBERT, XLM), Retrieval-Augmented Generation (RAG)  
🔹 **NLP Techniques:** Word Embeddings, Semantic Retrieval, Knowledge Graph Construction  
🔹 **Cloud & Deployment:** AWS, Azure  

---

## 📂 Dataset & Data Collection  
📌 **Data Sources:**  
- Collected **800 webpages** across **8 IAB Tier-1 categories**.  
- Included **150 multilingual URLs** (German, French, Chinese) for cross-language evaluation.  

📌 **Data Processing:**  
- Extracted **keywords and hierarchical structures** for **knowledge graph modeling**.  
- Used **search engines** to **retrieve top webpages** based on **taxonomy-defined keywords**.  

---

## 📌 Model Architecture  

### **🔹 Graph Construction (Neo4j)**  
- Built a **hierarchical knowledge graph** capturing **parent-child relationships**.  
- Modeled **synonyms, entity similarity, and contextual triplets** for better **semantic retrieval**.  

### **🔹 RAG-Based Classification**  
- Utilized **multiple LLM embeddings** (**mBERT, LaBSE, XLM, DistilBERT**) for **vector retrieval**.  
- Performed **similarity search** over **Neo4j’s hierarchical knowledge base**.  
- Implemented **ensemble voting** across **multiple models** to improve classification precision.  

### **🔹 Weight-Based Refinement**  
- Introduced **cluster-based scoring** to **rank synonyms & keywords** based on relevance.  
- Applied **min-max normalization** and **pruning** to remove noisy **misclassified nodes**.  

---

## 🔬 Results & Observations  

| Approach | Tier 1 Accuracy | Tier 2 Accuracy |
|----------|---------------|---------------|
| All Models (Baseline) | 47% | 26% |
| Excluding Biased Models | 53% | 30% |
| LLM Summarization | 65% | 40% |
| Summarized + Original Text (0.5 Weight) | 63% | 37% |
| **mBERT + LaBSE (Summarized)** | **70%** | **43%** |

📈 **Key Insights:**  
✔️ **Excluding biased models (RoBERTa, mDeBERTa)** improved performance.  
✔️ **LLM summarization** increased **classification accuracy by ~23%**.  
✔️ **Weight-based node refinement** further **reduced misclassification in hierarchical tiers**.  

---

## 🔮 Future Scope  
🔹 **Expand taxonomy classification** across **more IAB categories**.  
🔹 **Fine-tune LLMs & embeddings** for **domain-specific retrieval**.  
🔹 **Reduce computation cost** for OpenAI API-based **summarization techniques**.  
🔹 **Improve graph ranking models** using **advanced contextual embeddings**.  
