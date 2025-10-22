# Understanding Cohere Embedding Input Types

## Overview

When using Cohere embeddings for semantic search, you need to specify an `input_type` parameter. This guide explains what it means, why it matters, and how to use it correctly in your RAG (Retrieval-Augmented Generation) multilingual demo.

## The Core Concept: Asymmetric vs Symmetric Search

### Asymmetric Search (Our Use Case)

In asymmetric search, **queries** and **documents** are fundamentally different in nature:

- **Query**: Short, question-like, what the user types
  - Example: `"car won't start"`
  
- **Document**: Longer, detailed descriptions with context
  - Example: `"Customer reported that vehicle failed to start after being parked overnight in cold weather. Battery voltage measured at 11.2V. Recommended battery replacement and cold weather maintenance check."`

The model needs to understand that these different text types should match semantically, even though they're written very differently.

### Why Two Different Input Types?

Cohere trains embeddings to create **different vector representations** based on the `input_type`:

| Input Type | Purpose | When to Use |
|------------|---------|-------------|
| `search_document` | Optimized to encode context, details, and complete information. Creates vectors that are good at "answering" | When embedding documents to be stored in a search index |
| `search_query` | Optimized to encode intent, questions, and short phrases. Creates vectors that are good at "asking" | When embedding user queries to search against the index |

## Your Workflow Visualized

```
┌─────────────────────────────────────────────────────────┐
│  PHASE 1: INDEXING (indexing.ipynb)                     │
│                                                          │
│  Car Problems (Documents)                               │
│  "Engine overheating during highway driving..."         │
│                    ↓                                     │
│  Cohere Embed API                                       │
│  input_type="search_document"                           │
│                    ↓                                     │
│  Vector Embeddings [0.23, -0.45, 0.67, ...]            │
│                    ↓                                     │
│  Azure AI Search Index (Storage)                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE 2: SEARCH (query time)                           │
│                                                          │
│  User Query                                             │
│  "car overheating"                                      │
│                    ↓                                     │
│  Cohere Embed API                                       │
│  input_type="search_query"                              │
│                    ↓                                     │
│  Query Vector [0.19, -0.41, 0.71, ...]                 │
│                    ↓                                     │
│  Compare with Azure AI Search Index                     │
│  (Cosine Similarity / Vector Search)                    │
│                    ↓                                     │
│  Return Top Matching Car Problems                       │
└─────────────────────────────────────────────────────────┘
```

## Implementation Guide

### Phase 1: Indexing Documents (Current Notebook)

When embedding car problems to store in Azure AI Search:

```python
# Embedding documents for storage
response = co.embed(
    texts=[problem['fault'] for problem in problems],
    model='embed-multilingual-v3.0',
    input_type="search_document"  # ✓ Use this for indexing
)

# Extract embeddings
embeddings = response.embeddings
```

### Phase 2: Querying (Future Implementation)

When a user searches for car problems:

```python
# Embedding user query for search
response = co.embed(
    texts=[user_query],
    model='embed-multilingual-v3.0',
    input_type="search_query"  # ✓ Use this for queries
)

# Extract query embedding
query_embedding = response.embeddings[0]

# Use this to search Azure AI Search
# (Azure AI Search will compare this vector with stored document vectors)
```

## Other Input Types (For Reference)

While we use `search_document` and `search_query` for our semantic search use case, Cohere supports other input types:

| Input Type | Use Case | Example |
|------------|----------|---------|
| `classification` | Training or using text classifiers | Categorizing support tickets |
| `clustering` | Grouping similar items | Finding similar customer complaints |
| `image` | Embedding images (v3.0 only) | Image search applications |

## Why the Documentation Can Be Confusing

The Cohere documentation covers all use cases (search, classification, clustering) simultaneously, which can be overwhelming. Key issues:

1. **Not workflow-oriented**: Explains parameters but not the typical "indexing → searching" workflow
2. **Multiple use cases mixed**: Tries to explain everything at once
3. **Assumes prior knowledge**: Assumes familiarity with asymmetric search concepts

## Quick Reference for This Project

| What You're Doing | Input Type to Use | File/Phase |
|-------------------|-------------------|------------|
| Storing car problems in Azure AI Search | `search_document` | `indexing.ipynb` |
| User searching for car problems | `search_query` | Query implementation |
| Any other use case | Not applicable | N/A |

## Key Takeaway

Think of it like a conversation:
- **Documents** (`search_document`): These are the "answers" waiting in your database
- **Queries** (`search_query`): These are the "questions" users ask

By using different `input_type` values, Cohere optimizes the embeddings so that questions and answers can find each other, even when worded differently.

## Resources

- [Cohere Embed API Documentation](https://docs.cohere.com/docs/cohere-embed)
- [Cohere Multilingual Models](https://docs.cohere.com/docs/multilingual-language-models)
- [Azure AI Search Vector Search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview)