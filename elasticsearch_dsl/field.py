#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import base64
import collections.abc
import ipaddress
from copy import deepcopy
from datetime import date, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from dateutil import parser, tz

from .exceptions import ValidationException
from .query import Q
from .utils import AttrDict, AttrList, DslBase
from .wrappers import Range

if TYPE_CHECKING:
    from datetime import tzinfo
    from ipaddress import IPv4Address, IPv6Address

    from _operator import _SupportsComparison

    from .document import InnerDoc
    from .mapping_base import MappingBase
    from .query import Query

unicode = str


def construct_field(
    name_or_field: Union[
        str,
        "Field",
        Dict[str, Any],
    ],
    **params: Any,
) -> "Field":
    # {"type": "text", "analyzer": "snowball"}
    if isinstance(name_or_field, collections.abc.Mapping):
        if params:
            raise ValueError(
                "construct_field() cannot accept parameters when passing in a dict."
            )
        params = deepcopy(name_or_field)
        if "type" not in params:
            # inner object can be implicitly defined
            if "properties" in params:
                name = "object"
            else:
                raise ValueError('construct_field() needs to have a "type" key.')
        else:
            name = params.pop("type")
        return Field.get_dsl_class(name)(**params)

    # Text()
    if isinstance(name_or_field, Field):
        if params:
            raise ValueError(
                "construct_field() cannot accept parameters "
                "when passing in a construct_field object."
            )
        return name_or_field

    # "text", analyzer="snowball"
    return Field.get_dsl_class(name_or_field)(**params)


class Field(DslBase):
    _type_name = "field"
    _type_shortcut = staticmethod(construct_field)
    # all fields can be multifields
    _param_defs = {"fields": {"type": "field", "hash": True}}
    name = ""
    _coerce = False

    def __init__(
        self, multi: bool = False, required: bool = False, *args: Any, **kwargs: Any
    ):
        """
        :arg bool multi: specifies whether field can contain array of values
        :arg bool required: specifies whether field is required
        """
        self._multi = multi
        self._required = required
        super().__init__(*args, **kwargs)

    def __getitem__(self, subfield: str) -> "Field":
        return cast(Field, self._params.get("fields", {})[subfield])

    def _serialize(self, data: Any) -> Any:
        return data

    def _deserialize(self, data: Any) -> Any:
        return data

    def _empty(self) -> Optional[Any]:
        return None

    def empty(self) -> Optional[Any]:
        if self._multi:
            return AttrList([])
        return self._empty()

    def serialize(self, data: Any) -> Any:
        if isinstance(data, (list, AttrList, tuple)):
            return list(map(self._serialize, cast(Iterable[Any], data)))
        return self._serialize(data)

    def deserialize(self, data: Any) -> Any:
        if isinstance(data, (list, AttrList, tuple)):
            data = [
                None if d is None else self._deserialize(d)
                for d in cast(Iterable[Any], data)
            ]
            return data
        if data is None:
            return None
        return self._deserialize(data)

    def clean(self, data: Any) -> Any:
        if data is not None:
            data = self.deserialize(data)
        if data in (None, [], {}) and self._required:
            raise ValidationException("Value required for this field.")
        return data

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        name, value = cast(Tuple[str, Dict[str, Any]], d.popitem())
        value["type"] = name
        return value


class CustomField(Field):
    name = "custom"
    _coerce = True

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.builtin_type, Field):
            return self.builtin_type.to_dict()

        d = super().to_dict()
        d["type"] = self.builtin_type
        return d


class Object(Field):
    name = "object"
    _coerce = True

    def __init__(
        self,
        doc_class: Optional[Type["InnerDoc"]] = None,
        dynamic: Optional[Union[bool, str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        :arg document.InnerDoc doc_class: base doc class that handles mapping.
            If no `doc_class` is provided, new instance of `InnerDoc` will be created,
            populated with `properties` and used. Can not be provided together with `properties`
        :arg dynamic: whether new properties may be created dynamically.
            Valid values are `True`, `False`, `'strict'`.
            Can not be provided together with `doc_class`.
            See https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic.html
            for more details
        :arg dict properties: used to construct underlying mapping if no `doc_class` is provided.
            Can not be provided together with `doc_class`
        """
        if doc_class and (properties or dynamic is not None):
            raise ValidationException(
                "doc_class and properties/dynamic should not be provided together"
            )
        if doc_class:
            self._doc_class: Type["InnerDoc"] = doc_class
        else:
            # FIXME import
            from .document import InnerDoc

            # no InnerDoc subclass, creating one instead...
            self._doc_class = type("InnerDoc", (InnerDoc,), {})
            for name, field in (properties or {}).items():
                self._doc_class._doc_type.mapping.field(name, field)
            if dynamic is not None:
                self._doc_class._doc_type.mapping.meta("dynamic", dynamic)

        self._mapping: "MappingBase" = deepcopy(self._doc_class._doc_type.mapping)
        super().__init__(**kwargs)

    def __getitem__(self, name: str) -> Field:
        return self._mapping[name]

    def __contains__(self, name: str) -> bool:
        return name in self._mapping

    def _empty(self) -> "InnerDoc":
        return self._wrap({})

    def _wrap(self, data: Dict[str, Any]) -> "InnerDoc":
        return self._doc_class.from_es(data, data_only=True)

    def empty(self) -> Union["InnerDoc", AttrList[Any]]:
        if self._multi:
            return AttrList[Any]([], self._wrap)
        return self._empty()

    def to_dict(self) -> Dict[str, Any]:
        d = self._mapping.to_dict()
        d.update(super().to_dict())
        return d

    def _collect_fields(self) -> Iterator[Field]:
        return self._mapping.properties._collect_fields()

    def _deserialize(self, data: Any) -> "InnerDoc":
        # don't wrap already wrapped data
        if isinstance(data, self._doc_class):
            return data

        if isinstance(data, AttrDict):
            data = data._d_

        return self._wrap(data)

    def _serialize(
        self, data: Optional[Union[Dict[str, Any], "InnerDoc"]]
    ) -> Optional[Dict[str, Any]]:
        if data is None:
            return None

        # somebody assigned raw dict to the field, we should tolerate that
        if isinstance(data, collections.abc.Mapping):
            return data

        return data.to_dict()

    def clean(self, data: Any) -> Any:
        data = super().clean(data)
        if data is None:
            return None
        if isinstance(data, (list, AttrList)):
            for d in cast(Iterator["InnerDoc"], data):
                d.full_clean()
        else:
            data.full_clean()
        return data

    def update(self, other: Any, update_only: bool = False) -> None:
        if not isinstance(other, Object):
            # not an inner/nested object, no merge possible
            return

        self._mapping.update(other._mapping, update_only)


class Nested(Object):
    name = "nested"

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("multi", True)
        super().__init__(*args, **kwargs)


class Date(Field):
    name = "date"
    _coerce = True

    def __init__(
        self,
        default_timezone: Optional[Union[str, "tzinfo"]] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        :arg default_timezone: timezone that will be automatically used for tz-naive values
            May be instance of `datetime.tzinfo` or string containing TZ offset
        """
        if isinstance(default_timezone, str):
            self._default_timezone = tz.gettz(default_timezone)
        else:
            self._default_timezone = default_timezone
        super().__init__(*args, **kwargs)

    def _deserialize(self, data: Any) -> Union[datetime, date]:
        if isinstance(data, str):
            try:
                data = parser.parse(data)
            except Exception as e:
                raise ValidationException(
                    f"Could not parse date from the value ({data!r})", e
                )
            # we treat the yyyy-MM-dd format as a special case
            if hasattr(self, "format") and self.format == "yyyy-MM-dd":
                data = data.date()

        if isinstance(data, datetime):
            if self._default_timezone and data.tzinfo is None:
                data = data.replace(tzinfo=self._default_timezone)
            return data
        if isinstance(data, date):
            return data
        if isinstance(data, int):
            # Divide by a float to preserve milliseconds on the datetime.
            return datetime.utcfromtimestamp(data / 1000.0)

        raise ValidationException(f"Could not parse date from the value ({data!r})")


class Text(Field):
    _param_defs = {
        "fields": {"type": "field", "hash": True},
        "analyzer": {"type": "analyzer"},
        "search_analyzer": {"type": "analyzer"},
        "search_quote_analyzer": {"type": "analyzer"},
    }
    name = "text"
    
    def __init__(
        self,
        multi: bool = False, 
        required: bool = False, 
        analyzer: str = "standard",
        eager_global_ordinals: bool = False,
        fielddata: bool = False,
        fielddata_frequency_filter: Optional[dict] = None,
        fields: Optional[dict] = None,
        index: bool = True,
        index_options: str = "positions",
        index_prefixes: Optional[dict] = None,
        index_phrases: bool = False,
        norms: bool = True,
        position_increment_gap: int = 100,
        store: bool = False,
        search_analyzer: Optional[str] = None,
        search_quote_analyzer: Optional[str] = None,
        similarity: str = "BM25",
        term_vector: str = "no",
        meta: Optional[dict] = None
    ):
        """
        Initializes a TextField with comprehensive mapping parameters for full-text content indexing in Elasticsearch.

        :param analyzer: The analyzer which should be used for the text field, both at index-time and at search-time (unless overridden by the search_analyzer).
            Defaults to the default index analyzer, or the standard analyzer.
            see https://www.elastic.co/guide/en/elasticsearch/reference/current/analyzer.html
        :param eager_global_ordinals: Should global ordinals be loaded eagerly on refresh? Accepts true or false (default). 
            Enabling this is a good idea on fields that are frequently used for (significant) terms aggregations.
        :param fielddata: Can the field use in-memory fielddata for sorting, aggregations, or scripting? Accepts true or false (default).
        :param fielddata_frequency_filter: Expert settings which allow to decide which values to load in memory when fielddata is enabled. By default all values are loaded.
            eg {
                "min": 0.001, // values with a frequency lower than 0.001 will not be loaded.
                "max":0.1, //This means that if a value appears in more than 10% of the documents, it will not be loaded into memory. 
                            The goal is to avoid loading very frequent values that would take up a lot of memory but may not be very useful for sorting or aggregation.
                "min_segment_size":500 // This means that for segments smaller than 500 bytes, all field values will be loaded into memory. 
                            This is intended to increase efficiency when dealing with small segments, as selective loading of small segment data may not be as efficient as loading all values.
            }
        :param fields: Multi-fields allow the same string value to be indexed in multiple ways for different purposes,
            such as one field for search and a multi-field for sorting and aggregations, or the same string value analyzed by different analyzers.
            eg: {
                    "raw": { 
                        "type":  "keyword"
                    }
                }
            see: https://www.elastic.co/guide/en/elasticsearch/reference/current/multi-fields.html
        :param index: Should the field be searchable? Accepts true (default) or false.
        :param index_options: The index_options parameter controls what information is added to the inverted index for search and highlighting purposes. Only term-based field types like text and keyword support this configuration.
            The parameter accepts one of the following values. Each value retrieves information from the previous listed values. For example, freqs contains docs; positions contains both freqs and docs.
            - docs: Only the doc number is indexed. Can answer the question "Does this term exist in this field?"
            - freqs: Doc number and term frequencies are indexed. Term frequencies are used to score repeated terms higher than single terms.
            - positions (default): Doc number, term frequencies, and term positions (or order) are indexed. Positions can be used for proximity or phrase queries.
            - offsets: Doc number, term frequencies, positions, and start and end character offsets (which map the term back to the original string) are indexed. Offsets 

        :param index_prefixes: The index_prefixes parameter enables the indexing of term prefixes to speed up prefix searches. It accepts the following optional settings:
            - min_chars: The minimum prefix length to index. Must be greater than 0, and defaults to 2. The value is inclusive.
            - max_chars: The maximum prefix length to index. Must be less than 20, and defaults to 5. The value is inclusive.
            eg: 
                {
                    "min_chars" : 1,
                    "max_chars" : 10
                }
        :param index_phrases: If enabled, two-term word combinations (shingles) are indexed into a separate field.
            This allows exact phrase queries (no slop) to run more efficiently, at the expense of a larger index. 
            Note that this works best when stopwords are not removed, 
            as phrases containing stopwords will not use the subsidiary field and will fall back to a standard phrase query.
            Accepts true or false (default).
        :param norms: Whether field-length should be taken into account when scoring queries. Accepts true (default) or false.
        :param position_increment_gap: The number of fake term position which should be inserted between each element of an array of strings. 
            Defaults to the position_increment_gap configured on the analyzer which defaults to 100. 
            100 was chosen because it prevents phrase queries with reasonably large slops (less than 100) from matching terms across field values.
        :param store: Whether the field value should be stored and retrievable separately from the _source field.
            Accepts true or false (default). This parameter will be automatically set to true for TSDB indices (indices that have index.mode set to time_series) 
            if there is no keyword sub-field that supports synthetic _source.
        :param search_analyzer: The analyzer that should be used at search time on the text field. Defaults to the analyzer setting.
        :param search_quote_analyzer: The analyzer that should be used at search time when a phrase is encountered. Defaults to the search_analyzer setting.
        :param similarity: Which scoring algorithm or similarity should be used. Defaults to BM25.
            -`BM25` The Okapi BM25 algorithm. The algorithm used by default in Elasticsearch and Lucene.
            -`boolean` A simple boolean similarity, which is used when full-text ranking is not needed and the score should only be based on whether the query terms match or not.
                Boolean similarity gives terms a score equal to their query boost.
            And custom similarity ,see https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html
        :param term_vector: Whether term vectors should be stored for the field. Defaults to no.
            - no: No term vectors are stored. (default)
            - yes: Just the terms in the field are stored.
            - with_positions: Terms and positions are stored.
            - with_offsets: Terms and character offsets are stored.
            - with_positions_offsets: Terms, positions, and character offsets are stored.
            - with_positions_payloads: Terms, positions, and payloads are stored.
            - with_positions_offsets_payloads: Terms, positions, offsets and payloads are stored.
            
            
            see https://www.elastic.co/guide/en/elasticsearch/reference/current/term-vector.html
        :param meta: Metadata about the field.
        """
        if fielddata and not fielddata_frequency_filter:
            raise ValueError("fielddata_frequency_filter must be provided when fielddata is enabled.")

        if index_prefixes:
            if 'min_chars' in index_prefixes and (index_prefixes['min_chars'] < 1 or index_prefixes['min_chars'] > 20):
                raise ValueError("min_chars in index_prefixes must be between 1 and 20.")
            if 'max_chars' in index_prefixes and (index_prefixes['max_chars'] < 1 or index_prefixes['max_chars'] > 20):
                raise ValueError("max_chars in index_prefixes must be between 1 and 20.")
            if 'min_chars' in index_prefixes and 'max_chars' in index_prefixes and index_prefixes['min_chars'] > index_prefixes['max_chars']:
                raise ValueError("min_chars cannot be greater than max_chars in index_prefixes.")

        if index_options not in ["docs", "freqs", "positions", "offsets"]:
            raise ValueError("Invalid value for index_options. Choose from 'docs', 'freqs', 'positions', or 'offsets'.")

        if term_vector not in ["no", "yes", "with_positions", "with_offsets", "with_positions_offsets", "with_positions_payloads", "with_positions_offsets_payloads"]:
            raise ValueError(
                "Invalid value for term_vector. Choose from 'no', 'yes', 'with_positions', 'with_offsets', 'with_positions_offsets', 'with_positions_payloads', or 'with_positions_offsets_payloads'.")



        field_parameters = {
            "multi": multi, 
            "required": required, 
            'analyzer': analyzer,
            'eager_global_ordinals': eager_global_ordinals,
            'fielddata': fielddata,
            'fielddata_frequency_filter': fielddata_frequency_filter,
            'fields': fields,
            'index': index,
            'index_options': index_options,
            'index_prefixes': index_prefixes,
            'index_phrases': index_phrases,
            'norms': norms,
            'position_increment_gap': position_increment_gap,
            'store': store,
            'search_analyzer': search_analyzer,  
            'search_quote_analyzer': search_quote_analyzer ,  
            'similarity': similarity,
            'term_vector': term_vector,
            'meta': meta
        }
        field_parameters = {k: v for k, v in field_parameters.items() if v is not None}
        super().__init__( **field_parameters)
    


class SearchAsYouType(Field):
    _param_defs = {
        "analyzer": {"type": "analyzer"},
        "search_analyzer": {"type": "analyzer"},
        "search_quote_analyzer": {"type": "analyzer"},
    }
    name = "search_as_you_type"


class Keyword(Field):
    _param_defs = {
        "fields": {"type": "field", "hash": True},
        "search_analyzer": {"type": "analyzer"},
        "normalizer": {"type": "normalizer"},
    }
    name = "keyword"
    def __init__(
        self,
        multi: bool = False, 
        required: bool = False, 
        ignore_above: int = 256,
        doc_values: bool = True,
        eager_global_ordinals: bool = False,
        fields: dict = None,
        index: bool = True,
        index_options: str = "docs",
        meta: dict = None,
        norms: bool = False,
        null_value: str = None,
        on_script_error: str = "fail",
        script: str = None,
        store: bool = False,
        similarity: str = "BM25",
        normalizer: str = None,
        split_queries_on_whitespace: bool = False,
        time_series_dimension: bool = False
    ):
        """
        :param ignore_above: Do not index any string longer than this value. Defaults to 2147483647 so that all values would be accepted. Please however note that default dynamic mapping rules create a sub keyword field that overrides this default by setting ignore_above: 256.
        :param doc_values: Should the field be stored on disk in a column-stride fashion, so that it can later be used for sorting, aggregations, or scripting? Accepts true (default) or false.
        :param eager_global_ordinals: Should global ordinals be loaded eagerly on refresh? Accepts true or false (default). Enabling this is a good idea on fields that are frequently used for terms aggregations.
        :param fields: Multi-fields allow the same string value to be indexed in multiple ways for different purposes, such as one field for search and a multi-field for sorting and aggregations.
        :param index: Should the field be quickly searchable? Accepts true (default) and false. keyword fields that only have doc_values enabled can still be queried, albeit slower.
        :param index_options: What information should be stored in the index, for scoring purposes. 
            Defaults to docs but can also be set to freqs to take term frequency into account when computing scores.
            - docs: Only the doc number is indexed. Can answer the question "Does this term exist in this field?"
            - freqs: Doc number and term frequencies are indexed. Term frequencies are used to score repeated terms higher than single terms.
            - positions (default): Doc number, term frequencies, and term positions (or order) are indexed. Positions can be used for proximity or phrase queries.
            - offsets: Doc number, term frequencies, positions, and start and end character offsets (which map the term back to the original string) are indexed. Offsets 
        :param meta: Metadata about the field.
        :param norms: Whether field-length should be taken into account when scoring queries. Accepts true or false (default).
        :param null_value: Accepts a string value which is substituted for any explicit null values. Defaults to null, which means the field is treated as missing. Note that this cannot be set if the script value is used.
        :param on_script_error: Defines what to do if the script defined by the script parameter throws an error at indexing time. Accepts fail (default), which will cause the entire document to be rejected, and continue, which will register the field in the document’s _ignored metadata field and continue indexing. This parameter can only be set if the script field is also set.
        :param script: If this parameter is set, then the field will index values generated by this script, rather than reading the values directly from the source. If a value is set for this field on the input document, then the document will be rejected with an error. Scripts are in the same format as their runtime equivalent. Values emitted by the script are normalized as usual, and will be ignored if they are longer that the value set on ignore_above.
        :param store: Whether the field value should be stored and retrievable separately from the _source field. Accepts true or false (default).
        :param similarity: Which scoring algorithm or similarity should be used. Defaults to BM25.
        :param normalizer: How to pre-process the keyword prior to indexing. Defaults to null, meaning the keyword is kept as-is.
        :param split_queries_on_whitespace: Whether full text queries should split the input on whitespace when building a query for this field. Accepts true or false (default).
        :param time_series_dimension: Marks the field as a time series dimension. Defaults to false.

            The index.mapping.dimension_fields.limit index setting limits the number of dimensions in an index.

            Dimension fields have the following constraints:

            The doc_values and index mapping parameters must be true.
            Field values cannot be an array or multi-value.
            Dimension values are used to identify a document’s time series. If dimension values are altered in any way during indexing, the document will be stored as belonging to different from intended time series. As a result there are additional constraints:

            The field cannot use a normalizer.
        """
        if script and null_value is not None:
            raise ValueError("null_value cannot be set if script is provided.")
        
        if on_script_error != "fail" and script is None:
            raise ValueError("on_script_error can only be set if script is also set.")

        if time_series_dimension:
            if not doc_values or not index:
                raise ValueError("doc_values and index must be True for time series dimension fields.")
            if normalizer is not None:
                raise ValueError("normalizer must be None for time series dimension fields.")
            if isinstance(fields, (list, tuple)):
                raise ValueError("Field values cannot be an array or multi-value for time series dimension fields.")

        field_parameters = {
            "multi": multi, 
            "required": required, 
            'ignore_above': ignore_above,
            'doc_values': doc_values,
            'eager_global_ordinals': eager_global_ordinals,
            'fields': fields,
            'index': index,
            'index_options': index_options,
            'meta': meta,
            'norms': norms,
            'null_value': null_value,
            'on_script_error': on_script_error,
            'script': script,
            'store': store,
            'similarity': similarity,
            'normalizer': normalizer,
            'split_queries_on_whitespace': split_queries_on_whitespace,
            'time_series_dimension': time_series_dimension
        }
        field_parameters = {k: v for k, v in field_parameters.items() if v is not None}
        super().__init__( **field_parameters)


class ConstantKeyword(Keyword):
    name = "constant_keyword"


class Boolean(Field):
    name = "boolean"
    _coerce = True

    def _deserialize(self, data: Any) -> bool:
        if data == "false":
            return False
        return bool(data)

    def clean(self, data: Any) -> Optional[bool]:
        if data is not None:
            data = self.deserialize(data)
        if data is None and self._required:
            raise ValidationException("Value required for this field.")
        return data  # type: ignore


class Float(Field):
    name = "float"
    _coerce = True

    def _deserialize(self, data: Any) -> float:
        return float(data)


class DenseVector(Float):
    name = "dense_vector"
    def __init__(
        self, 
        dims: Optional[int] = None, 
        element_type: str = 'float',
        index: bool = True, 
        similarity: Optional[str] = None,
        index_options_type: str = "hnsw",
        index_options_m: int = 16,
        index_options_ef_construction:int=100,
        index_options_confidence_interval:float=None
        ):
        """
        Initializes a DenseVectorField with comprehensive mapping parameters tailored for kNN searches in Elasticsearch.

        :param dims: (Optional, integer) Number of vector dimensions, cannot exceed 4096. If not specified, it defaults 
                     to the length of the first vector added to the field.
        :param element_type: (Optional, string) 
            float
                indexes a 4-byte floating-point value per dimension. This is the default value.
            byte
                indexes a 1-byte integer value per dimension.
            bit
                indexes a single bit per dimension. Useful for very high-dimensional vectors or models that specifically support bit vectors. 
                NOTE: when using bit, the number of dimensions must be a multiple of 8 and must represent the number of bits.
        :param index: (Optional, boolean) If true, allows searching this field using the kNN search API. Defaults to true.
        :param similarity: (Optional, string) The vector similarity metric to use in kNN search. Defaults to 'l2_norm' 
                    for 'bit' element_type; otherwise, defaults to 'cosine'. The similarity metric determines how
                    document scores are computed based on their proximity to the query vector. Valid values include:
            l2_norm
                Computes similarity based on the L2 distance (also known as Euclidean distance) between the vectors. 
                The document _score is computed as 1 / (1 + l2_norm(query, vector)^2).
                For bit vectors, instead of using l2_norm, the hamming distance between the vectors is used. 
                The _score transformation is (numBits - hamming(a, b)) / numBits.
            dot_product
                Computes the dot product of two unit vectors. This option provides an optimized way to perform cosine similarity. 
                The constraints and computed score are defined by element_type.
                When element_type is float, all vectors must be unit length, including both document and query vectors. 
                The document _score is computed as (1 + dot_product(query, vector)) / 2.
                When element_type is byte, all vectors must have the same length including both document and query vectors 
                or results will be inaccurate. The document _score is computed as 0.5 + (dot_product(query, vector) / (32768 * dims)) 
                where dims is the number of dimensions per vector.
            cosine
                Computes the cosine similarity. During indexing Elasticsearch automatically normalizes vectors with cosine similarity 
                to unit length. This allows to internally use dot_product for computing similarity, which is more efficient. 
                Original un-normalized vectors can be still accessed through scripts. The document _score is computed as 
                (1 + cosine(query, vector)) / 2. The cosine similarity does not allow vectors with zero magnitude, 
                since cosine is not defined in this case.
            max_inner_product
                Computes the maximum inner product of two vectors. This is similar to dot_product, but doesn’t require vectors 
                to be normalized. This means that each vector’s magnitude can significantly affect the score. 
                The document _score is adjusted to prevent negative values. For max_inner_product values < 0, the _score is 
                1 / (1 + -1 * max_inner_product(query, vector)). For non-negative max_inner_product results the _score is 
                calculated max_inner_product(query, vector) + 1.
        :param index_options_type: (Optional, dict) Configures the kNN indexing algorithm. Includes settings such as:
            hnsw - This utilizes the HNSW algorithm for scalable approximate kNN search. This supports all element_type values.
            int8_hnsw - The default index type for float vectors. This utilizes the HNSW algorithm in addition to automatically scalar quantization for scalable approximate kNN search with element_type of float. This can reduce the memory footprint by 4x at the cost of some accuracy. See Automatically quantize vectors for kNN search.
            int4_hnsw - This utilizes the HNSW algorithm in addition to automatically scalar quantization for scalable approximate kNN search with element_type of float. This can reduce the memory footprint by 8x at the cost of some accuracy. See Automatically quantize vectors for kNN search.
            flat - This utilizes a brute-force search algorithm for exact kNN search. This supports all element_type values.
            int8_flat - This utilizes a brute-force search algorithm in addition to automatically scalar quantization. Only supports element_type of float.
            int4_flat - This utilizes a brute-force search algorithm in addition to automatically half-byte scalar quantization. Only supports element_type of float.
        :param index_options_m: (Optional, integer) The number of neighbors each node will be connected to in the HNSW graph. Defaults to 16. Only applicable to hnsw, int8_hnsw, and int4_hnsw index types. 
        :param index_options_ef_construction: (Optional, integer)  The number of candidate nodes tracked while assembling the nearest neighbors during the index construction. Defaults to 100.
        :param index_options_confidence_interval: (Optional, float) Only applicable to int8_hnsw, int4_hnsw, int8_flat, and int4_flat index types.
                The confidence interval to use when quantizing the vectors.
                Can be any value between and including 0.90 and 1.0 or exactly 0.
                When the value is 0, this indicates that dynamic quantiles should be calculated for optimized quantization.
                When between 0.90 and 1.0, this value restricts the values used when calculating the quantization thresholds. 
                For example, a value of 0.95 will only use the middle 95% of the values when calculating the quantization thresholds 
                (e.g. the highest and lowest 2.5% of values will be ignored).
                Defaults to 1/(dims + 1) for int8 quantized vectors and 0 for int4 for dynamic quantile calculation.

        """
        # Default index options if none provided and indexing is enabled
        index_options = {
            'type': index_options_type,
            'm': index_options_m,  # Number of neighbors each node will be connected to in the HNSW graph
            'ef_construction': index_options_ef_construction
        }
        
        # Validate similarity
        valid_similarities = {'l2_norm', 'dot_product', 'cosine', 'max_inner_product'}
        assert similarity is None or similarity in valid_similarities, f"similarity must be one of {valid_similarities}"
        assert element_type in {'float', 'byte', 'bit'}, "element_type must be one of 'float', 'byte', 'bit'"
        if similarity is None:
            similarity = 'l2_norm' if element_type == 'bit' else 'cosine'  
            
        if index_options_confidence_interval is not None :
            if index_options_type in {'int8_hnsw', 'int4_hnsw', 'int8_flat', 'int4_flat'}:
                assert isinstance(index_options_confidence_interval, float) and 0<=index_options_confidence_interval<=1,\
                    "index_options 'confidence_interval' must be a float between 0 and 1"
                index_options["confidence_interval"] = index_options_confidence_interval
        # Validate index_options
        valid_types = {'hnsw', 'int8_hnsw', 'int4_hnsw', 'flat', 'int8_flat', 'int4_flat'}
        assert 'type' in index_options and index_options['type'] in valid_types, f"index_options 'type' must be one of {valid_types}"
        assert 'm' not in index_options or isinstance(index_options['m'], int), "index_options 'm' must be an integer"
        assert 'ef_construction' not in index_options or isinstance(index_options['ef_construction'], int), "index_options 'ef_construction' must be an integer"
        
        field_parameters = {
            "multi": True,
            'dims': dims,
            'element_type': element_type,
            'index': index,
            'similarity': similarity or ('l2_norm' if element_type == 'bit' else 'cosine'),
            'index_options': index_options if index else None
        }
        field_parameters = {k: v for k, v in field_parameters.items() if v is not None}
        super().__init__( **field_parameters)
    


class SparseVector(Field):
    name = "sparse_vector"


class HalfFloat(Float):
    name = "half_float"


class ScaledFloat(Float):
    name = "scaled_float"

    def __init__(self, scaling_factor: int, *args: Any, **kwargs: Any):
        super().__init__(scaling_factor=scaling_factor, *args, **kwargs)


class Double(Float):
    name = "double"


class RankFeature(Float):
    name = "rank_feature"


class RankFeatures(Field):
    name = "rank_features"


class Integer(Field):
    name = "integer"
    _coerce = True

    def _deserialize(self, data: Any) -> int:
        return int(data)


class Byte(Integer):
    name = "byte"


class Short(Integer):
    name = "short"


class Long(Integer):
    name = "long"


class Ip(Field):
    name = "ip"
    _coerce = True

    def _deserialize(self, data: Any) -> Union["IPv4Address", "IPv6Address"]:
        # the ipaddress library for pypy only accepts unicode.
        return ipaddress.ip_address(unicode(data))

    def _serialize(self, data: Any) -> Optional[str]:
        if data is None:
            return None
        return str(data)


class Binary(Field):
    name = "binary"
    _coerce = True

    def clean(self, data: str) -> str:
        # Binary fields are opaque, so there's not much cleaning
        # that can be done.
        return data

    def _deserialize(self, data: Any) -> bytes:
        return base64.b64decode(data)

    def _serialize(self, data: Any) -> Optional[str]:
        if data is None:
            return None
        return base64.b64encode(data).decode()


class GeoPoint(Field):
    name = "geo_point"


class GeoShape(Field):
    name = "geo_shape"


class Completion(Field):
    _param_defs = {
        "analyzer": {"type": "analyzer"},
        "search_analyzer": {"type": "analyzer"},
    }
    name = "completion"


class Percolator(Field):
    name = "percolator"
    _coerce = True

    def _deserialize(self, data: Any) -> "Query":
        return Q(data)  # type: ignore

    def _serialize(self, data: Any) -> Optional[Dict[str, Any]]:
        if data is None:
            return None
        return data.to_dict()  # type: ignore


class RangeField(Field):
    _coerce = True
    _core_field: Optional[Field] = None

    def _deserialize(self, data: Any) -> Range["_SupportsComparison"]:
        if isinstance(data, Range):
            return data
        data = {k: self._core_field.deserialize(v) for k, v in data.items()}  # type: ignore
        return Range(data)

    def _serialize(self, data: Any) -> Optional[Dict[str, Any]]:
        if data is None:
            return None
        if not isinstance(data, collections.abc.Mapping):
            data = data.to_dict()
        return {k: self._core_field.serialize(v) for k, v in data.items()}  # type: ignore


class IntegerRange(RangeField):
    name = "integer_range"
    _core_field = Integer()


class FloatRange(RangeField):
    name = "float_range"
    _core_field = Float()


class LongRange(RangeField):
    name = "long_range"
    _core_field = Long()


class DoubleRange(RangeField):
    name = "double_range"
    _core_field = Double()


class DateRange(RangeField):
    name = "date_range"
    _core_field = Date()


class IpRange(Field):
    # not a RangeField since ip_range supports CIDR ranges
    name = "ip_range"


class Join(Field):
    name = "join"


class TokenCount(Field):
    name = "token_count"


class Murmur3(Field):
    name = "murmur3"


class SemanticText(Field):
    name = "semantic_text"
