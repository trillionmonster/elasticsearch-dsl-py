
"""
 Single Field Operations and Complex Search 

This script is designed to test and demonstrate advanced search queries and single-field operations using Elasticsearch.

Key Features:
- Index creation with data populated from Cartesian products.
- Execution of complex DSL queries, including KNN searches and geospatial queries.
- Addition and updating of single fields in documents, both synchronously and asynchronously.
- Integration with embedding services to enhance search capabilities.

Usage:
- Customize the ES_CONFIG with your Elasticsearch settings.
- Run the desired function by uncommenting it in the main block.
"""

from typing import List
from elasticsearch_dsl import (
    Document,
    Date,
    Integer,
    Search,
    AsyncSearch,
    Text,
    Keyword,
    DenseVector,
    GeoPoint,
    connections,
    async_connections,
    Index,
    Embedding,
    SingleField,
    SingleFieldValue,
    AsyncSingleField,
)
from elasticsearch import AsyncElasticsearch
from itertools import product
from datetime import date
from tqdm import tqdm
import re
from elasticsearch_dsl.query import Knn, Match, Regexp, Term, GeoDistance
import os

# Initialize embedder instances
embedder_1024 = Embedding(base_url='http://localhost:8083', model_name="/Conan")
embedder_512 = Embedding(
    base_url='http://localhost:8080',
    model_name="/stella-mrl-large-zh-v3.5-1792d",
    select_dims=512
)

# Elasticsearch configuration
ES_CONFIG = {
    "address": "http://127.0.0.1:9200",
    "user": "elastic",
    "password": os.environ.get("ES_PASSWORD", "Moving9527135246"),
    "index_name": "cartesian_docs"
}

class CartesianDoc(Document):
    """
    Elasticsearch document mapping for Cartesian products.
    """
    A = Integer()
    B = Integer()
    C = Text()
    D = Keyword()
    E = DenseVector(dims=512)
    F = DenseVector(dims=1024)
    G = Date()
    location = GeoPoint()
    city = Keyword()

    class Index:
        name = ES_CONFIG["index_name"]
        # Index settings
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }

    def save(self, **kwargs):
        """
        Save the document to Elasticsearch.
        """
        return super().save(**kwargs)

def create_index():
    """
    Create Elasticsearch index and populate it with Cartesian product documents.
    """
    # Establish connection to Elasticsearch
    connections.create_connection(
        hosts=[ES_CONFIG["address"]],
        http_auth=(ES_CONFIG["user"], ES_CONFIG["password"]),
        timeout=300
    )

    # Delete the index if it exists and create a new one
    if CartesianDoc._index.exists():
        Index(ES_CONFIG["index_name"]).delete()
    CartesianDoc.init()

    # City coordinates
    CITIES = {
        "北京": {"lat": 39.9042, "lon": 116.4074},
        "上海": {"lat": 31.2304, "lon": 121.4737},
        "深圳": {"lat": 22.5431, "lon": 114.0579}
    }

    # Time expressions
    TIME_WORDS = {
        "今天": "today",
        "明天": "tomorrow",
        "后天": "the day after tomorrow"
    }

    doc_id = 0
    A_values = [1, 2]
    B_values = [1, 2]
    D_values = [
        'A/a/1', 'A/a/2', 'A/b/1', 'A/b/2',
        'B/a/1', 'B/a/2', 'B/b/1', 'B/b/2'
    ]
    G_values = [date(2023, 10, 7), date(2023, 10, 8)]
    C_values = []

    # Generate text content for field C
    for city_zh, city_en in [("北京", "Beijing"), ("上海", "Shanghai"), ("深圳", "Shenzhen")]:
        # Weather-related questions
        for zh_time, en_time in TIME_WORDS.items():
            C_values.append(f"{zh_time}{city_zh}天气如何")
            C_values.append(f"How is the weather in {city_en} {en_time}")

        # Other questions
        C_values.extend([
            f"{city_zh}房价如何",
            f"How is the housing price in {city_en}",
            f"{city_zh}穿什么衣服",
            f"What to wear in {city_en}"
        ])

    docs: List[CartesianDoc] = []

    # Generate Cartesian products and create documents
    for A, B, C_text, D, G in tqdm(list(product(A_values, B_values, C_values, D_values, G_values))):
        # Determine city based on text content
        city = next((city for city in CITIES.keys() if city in C_text), None)
        if city is None:
            city = next((city for city in CITIES.keys() if city.lower() in C_text.lower()), "北京")

        # Create document instance
        doc = CartesianDoc(
            meta={'id': doc_id},
            A=A,
            B=B,
            C=C_text,
            D=D,
            E=embedder_512.embedding(C_text)[0],
            F=embedder_1024.embedding(C_text)[0],
            G=G,
            location=CITIES[city],
            city=city
        )
        docs.append(doc)
        doc_id += 1

    # Bulk insert documents into Elasticsearch
    CartesianDoc.bulk(actions=docs, chunk_size=1000)
    print("All documents have been inserted.")

def dsl_query():
    """
    Perform a DSL query on the Elasticsearch index.
    """
    # Establish connection to Elasticsearch
    connections.create_connection(
        hosts=[ES_CONFIG["address"]],
        http_auth=(ES_CONFIG["user"], ES_CONFIG["password"]),
        timeout=300
    )

    # Initialize embedder
    embedder_1024 = Embedding(base_url='http://localhost:8083', model_name="/Conan")

    # Generate embedding for query term
    embed = embedder_1024.embedding(["衣服"])[0]

    # Define individual query components
    q1 = Term("A", 1)
    q2 = Match("C", "深圳")
    q4 = Regexp("D", "A/.*/2")
    q5 = Term("B", 2)
    q6 = GeoDistance(
        _field="location",
        _value={"lat": 22.5430, "lon": 114.0500},
        distance="10km"
    )
    q3 = Knn(field="HH", query_vector=embed, k=4, num_candidates=10000)

    # Combine queries using Boolean operators
    q = q3 ** (q1 & ((q5 % ~q4) & q2) & q6)

    # Execute the search query
    s = Search().query(q).index(ES_CONFIG["index_name"]).source(["A", "B", "C", "D", "G"])
    hits = s.execute()

    # Process and validate the results
    for h in hits:
        obj = h.to_dict()
        assert h.meta.id is not None
        assert obj["A"] == 1
        assert "深圳" in obj["C"]
        assert re.match(r"A/.*/2", obj["D"]) is None
        assert obj["B"] == 2

def single_field_operate():
    """
    Add or update a single field in Elasticsearch documents.
    """
    # Establish connection to Elasticsearch
    connections.create_connection(
        hosts=[ES_CONFIG["address"]],
        http_auth=(ES_CONFIG["user"], ES_CONFIG["password"]),
        timeout=300
    )

    # Initialize embedder
    embedder_1024 = Embedding(base_url='http://localhost:8083', model_name="/Conan")

    # Initialize SingleField instance
    single_field = SingleField(
        index_name=ES_CONFIG["index_name"],
        field_name="HH",
        field_type=DenseVector(dims=1024)
    )

    # Retrieve documents to update
    search_query = Search(index=ES_CONFIG["index_name"]).source(includes=['C'])
    docs = list(search_query.scan())

    # Generate embeddings for field 'C'
    embeddings = embedder_1024.embedding([item.C for item in docs], verbose=True)
    datas = []

    # Prepare data for bulk update
    for doc, embedding_vector in zip(docs, embeddings):
        datas.append(SingleFieldValue(embedding_vector, doc.meta.id))

    # Perform bulk update of the single field
    result = single_field.bulk(datas, verbose=True, total=len(datas), chunk_size=10, stats_only=True)
    print(result)

async def async_single_field_operate():
    """
    Asynchronously add or update a single field in Elasticsearch documents.
    """
    # Establish asynchronous connection to Elasticsearch
    async_connections.create_connection(
        hosts=[ES_CONFIG["address"]],
        http_auth=(ES_CONFIG["user"], ES_CONFIG["password"]),
        timeout=300
    )

    # Initialize embedder
    embedder_1024 = Embedding(base_url='http://localhost:8083', model_name="/Conan")

    # Initialize AsyncSingleField instance
    single_field = AsyncSingleField(
        index_name=ES_CONFIG["index_name"],
        field_name="HH",
        field_type=DenseVector(dims=1024)
    )
    await single_field.put_mapping()

    # Execute asynchronous search query
    search_query = AsyncSearch(index=ES_CONFIG["index_name"]).source(includes=['C'])
    docs = []
    async for doc in search_query.scan():
        docs.append(doc)

    # Generate embeddings for field 'C'
    embeddings = embedder_1024.embedding([item.C for item in docs], verbose=True)
    datas = []

    # Prepare data for bulk update
    for doc, embedding_vector in zip(docs, embeddings):
        datas.append(SingleFieldValue(embedding_vector, doc.meta.id))

    # Perform bulk update asynchronously
    result = await single_field.bulk(datas, verbose=True, total=len(datas), chunk_size=10, stats_only=True)
    print(result)

if __name__ == '__main__':
    # Uncomment the function you want to run
    create_index()
    
    single_field_operate()
    dsl_query()
    
    import asyncio
    asyncio.run(async_single_field_operate())
    dsl_query()
    
