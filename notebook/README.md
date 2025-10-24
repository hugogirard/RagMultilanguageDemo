# 🚗 Multilingual RAG Demo - Car Troubleshooting System

## 🎯 Project Goal

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system for multilingual car troubleshooting. The goal is to explore different strategies for handling multilingual search and retrieval in AI applications, specifically comparing:

- **Native language embeddings** vs. **translated embeddings**
- **Single vector** vs. **dual vector** approaches
- Performance across 7 languages: English, French, Spanish, Japanese, Chinese, Greek, and Hebrew

The system helps users find car problem solutions by searching through a knowledge base in their native language, even when the stored data was originally written in different languages.

---

## 📓 Notebooks Overview

Execute the notebooks in the following order:

### 1️⃣ `generate.data.ipynb` - Dataset Generation

**Purpose**: Creates a realistic multilingual car troubleshooting dataset.

**What it does**:
- Generates car problems and solutions across **5 brands** and **10 models**
- Covers **3 common car issues**: engine overheating, brake noise, battery drain
- Produces content in **7 languages** with natural variations
- Includes intentional grammatical variations to simulate real user input
- Exports data to `car_problems_multilingual.xlsx`

**When to run**: Execute this notebook first to generate the test data.

**Output**: `car_problems_multilingual.xlsx` - 60 records with multilingual car problems and solutions

---

### 2️⃣ `create-index.ipynb` - Azure AI Search Index Setup

**Purpose**: Creates three different Azure AI Search indexes to test multilingual retrieval strategies.

**What it does**:
- Configures three index schemas with different multilingual approaches:
  - **`multilanguage`**: Vectorizes content in original language (Cohere 1024-dim)
  - **`translated`**: Translates to English before vectorizing (OpenAI 1536-dim)
  - **`translated_dual`**: Stores both original and English vectors (hybrid approach)
- Sets up HNSW vector search for efficient semantic matching
- Configures faceted navigation and filtering capabilities

**When to run**: Execute this notebook after generating the data, before indexing documents.

**Prerequisites**:
- Azure AI Search service
- Environment variables configured in `.env`:
  ```
  SEARCH_ENDPOINT=https://your-service.search.windows.net
  SEARCH_API_KEY=your-admin-api-key
  ```

**Output**: Three search indexes ready for document ingestion and testing

---

## 🔄 Workflow

```
┌─────────────────────────┐
│  1. generate.data.ipynb │  ──→  Generate multilingual dataset
└─────────────────────────┘
            │
            ↓
┌─────────────────────────┐
│  2. create-index.ipynb  │  ──→  Create Azure AI Search indexes
└─────────────────────────┘
            │
            ↓
     [Next Steps]
     • Index documents
     • Query and test
     • Compare strategies
```

---

## 🧪 Testing Different Strategies

After running both notebooks, you can test and compare:

1. **Native Language Search**: Query in the user's language, retrieve from same language embeddings
2. **English Translation Search**: Query translated to English, retrieve from English embeddings
3. **Hybrid Search**: Leverage both native and English vectors for best results

---

## 📦 Requirements

Install the required packages:

```bash
pip install pandas openpyxl azure-search-documents python-dotenv
```

---

## 🚀 Getting Started

1. **Set up Azure AI Search**: Create a search service in Azure Portal
2. **Configure environment**: Create `.env` file with your credentials
3. **Run notebooks in order**:
   - First: `generate.data.ipynb`
   - Second: `create-index.ipynb`
4. **Index your data**: Load the generated Excel file into the indexes
5. **Query and compare**: Test search queries across different strategies

---

## 💡 Key Insights

This project helps answer:
- ✅ Should we translate all content to a single language for embeddings?
- ✅ Do multilingual embedding models work better than translation?
- ✅ Is there value in maintaining dual vectors (original + translated)?
- ✅ How does search quality vary across different languages?

---

## 📚 Learn More

- [Azure AI Search Documentation](https://learn.microsoft.com/azure/search/)
- [Vector Search in Azure AI Search](https://learn.microsoft.com/azure/search/vector-search-overview)
- [RAG Pattern](https://learn.microsoft.com/azure/search/retrieval-augmented-generation-overview)
