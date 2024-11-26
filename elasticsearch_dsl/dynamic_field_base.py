from typing import Any, Dict, List, Iterator
from .field import Field 
from dataclasses import dataclass


@dataclass
class SingleFieldValue:
    """
    Represents a single field value with an optional document ID.
    
    Attributes:
        value: The value to be stored in the field
        doc_id: Optional document ID for the Elasticsearch document
    """
    value: Any
    doc_id: str = None

@dataclass
class SingleFieldBase:
    """
    Base class for managing single field operations in Elasticsearch.
    
    Attributes:
        index_name: Name of the Elasticsearch index
        field_name: Name of the field
        field_type: Field type definition from elasticsearch_dsl
    """
    index_name: str
    field_name: str 
    field_type: Field

    

    def generate_dsl_to_bulk(self, value_list: List[SingleFieldValue]) -> Iterator[Dict]:
        """
        Generates Elasticsearch bulk update operations for a list of field values.
        
        Args:
            value_list: List of SingleFieldValue objects containing values and optional doc_ids
            
        Returns:
            Iterator of dictionaries containing Elasticsearch bulk update operations
            
        Raises:
            AssertionError: If value_list contains non-SingleFieldValue objects
        """
        assert all(isinstance(field_value, SingleFieldValue) for field_value in value_list), \
            "value_list must be a list of SingleFieldValue"
        
        for field_value in value_list:
            action = {
                "_op_type": "update",
                "_index": self.index_name,
                "doc": {self.field_name: field_value.value},
            }
            
            if field_value.doc_id:
                action["_id"] = field_value.doc_id
                
            yield action