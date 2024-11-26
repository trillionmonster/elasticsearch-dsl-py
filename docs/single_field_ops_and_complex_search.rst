.. _single_field_ops_and_complex_search:

Single Field Operations and Complex Search
========================================

This example demonstrates advanced search capabilities and single-field operations using elasticsearch-dsl. It shows how to efficiently manage and query Elasticsearch documents, with features like vector embeddings and complex DSL queries.

Key Features
-----------

* Index creation with Cartesian product data
* Complex DSL queries including KNN and geospatial searches  
* Single field operations (sync and async)
* Vector embedding integration
* Bulk document updates

Configuration
------------

The example uses the following configuration:

.. code-block:: python

    ES_CONFIG = {
        "address": "http://127.0.0.1:9200", 
        "user": "elastic",
        "password": os.environ.get("ES_PASSWORD"),
        "index_name": "cartesian_docs"
    }

Key Components
-------------

CartesianDoc
~~~~~~~~~~~~
A Document class that defines the mapping for Cartesian product documents in Elasticsearch.

create_index()
~~~~~~~~~~~~~
Creates an Elasticsearch index and populates it with documents generated from Cartesian products.

dsl_query() 
~~~~~~~~~~
Demonstrates complex query composition using DSL operators:

* KNN vector search
* Boolean combinations
* Regular expression matching
* Geospatial queries

single_field_operate()
~~~~~~~~~~~~~~~~~~~~
Shows how to update a single field across multiple documents:

1. Retrieves documents using Search
2. Generates embeddings for field values 
3. Performs bulk update using SingleField

async_single_field_operate()
~~~~~~~~~~~~~~~~~~~~~~~~~~
Asynchronous version of single_field_operate() using:

* AsyncSearch for document retrieval
* AsyncSingleField for bulk updates
* Async/await patterns

Usage
-----


1. Configure ES_CONFIG with your Elasticsearch settings

2. Run the desired operations:

* `Source Code <examples/single_field_ops_and_complex_search.py>`_

.. code-block:: python

    # Create index and documents
    create_index()
    
    # Run synchronous single field update
    single_field_operate()
    
    # Execute complex query
    dsl_query()
    
    # Run asynchronous single field update
    asyncio.run(async_single_field_operate())

Requirements
-----------

* elasticsearch-dsl
* Python 3.7+
* Elasticsearch 7.x+
* Running embedding service for vector operations


