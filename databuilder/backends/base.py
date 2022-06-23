import sqlalchemy

from databuilder.sqlalchemy_types import TYPES_BY_NAME, type_from_python_type

# Mutable global for storing registered backends
BACKENDS = {}


def register_backend(backend_class):
    BACKENDS[backend_class.backend_id] = backend_class


class BaseBackend:
    backend_id = None
    query_engine_class = None
    patient_join_column = None
    tables = None

    def __init_subclass__(cls, **kwargs):
        assert cls.backend_id is not None
        assert cls.query_engine_class is not None
        assert cls.patient_join_column is not None

        # Register each Backend by its id so we can identify it from an environment variable
        register_backend(cls)
        # Make sure each Backend knows what its tables are
        cls.tables = {}
        for name, value in vars(cls).items():
            if isinstance(value, SQLTable):
                cls._init_table(value)
                cls.tables[name] = value

    @classmethod
    def _init_table(cls, table):
        """
        Initialises the table
        Args:
            table: SQLTable Object
        """
        table.learn_patient_join(cls.patient_join_column)

    @classmethod
    def validate_contracts(cls):
        """
        Loops through all the tables in a backend and validates that
        each one meets any contract that it claims to implement
        """
        for name, table in cls.tables.items():
            contract = table.implements
            contract.validate_implementation(cls, name)

    def get_table_expression(self, table_name, schema):
        """
        Gets SQL expression for a table
        Args:
            table_name: Name of Table
            schema: a TableSchema
        Returns:
            A SQLAlchmey TableClause
        Raises:
            ValueError: If unknown table passed in
        """
        # TODO: We currently ignore the `schema` argument here. But I think we should
        # move towards having the supplied schema define the column types or, if not
        # that, then we should validate that the types match.
        print(self.tables)
        return self.tables[table_name].get_expression(table_name)


class SQLTable:
    def learn_patient_join(self, source):
        raise NotImplementedError()

    def _make_columns(self):
        return [
            self._make_column(name, column) for name, column in self.columns.items()
        ]

    def _make_column(self, name, column):
        source = column.source or name
        type_ = TYPES_BY_NAME[column.type].value
        sql_column = sqlalchemy.Column(source, type_, key=name)
        return sql_column


class MappedTable(SQLTable):
    def __init__(self, source, columns, implements=None, schema=None):
        self.source = source
        self.columns = columns
        self.implements = implements
        self._schema = schema

    def learn_patient_join(self, source):
        if "patient_id" not in self.columns:
            self.columns["patient_id"] = Column("integer", source)

    def get_expression(self, table_name):
        columns = self._make_columns()
        return sqlalchemy.table(self.source, *columns, schema=self._schema)


class QueryTable(SQLTable):
    def __init__(self, query, columns, implements=None, implementation_notes=None):
        self.query = query
        self.columns = columns
        self.implements = implements
        self.implementation_notes = implementation_notes or {}

    def learn_patient_join(self, source):
        # Given that `patient_id` has only one valid definition here, it should never be
        # explicitly specified
        assert "patient_id" not in self.columns
        self.columns["patient_id"] = Column("integer")

    def get_expression(self, table_name):
        columns = self._make_columns()
        query = sqlalchemy.text(self.query).columns(*columns)
        return query.alias(table_name)


class Column:
    def __init__(self, column_type, source=None):
        self.type = column_type
        self.source = source


class DefaultBackend:
    def get_table_expression(self, table_name, schema):
        """
        Returns a SQLAlchemy Table object matching the supplied name and schema
        """
        return sqlalchemy.table(
            table_name,
            sqlalchemy.Column("patient_id"),
            *[
                sqlalchemy.Column(name, type_=type_from_python_type(type_))
                for (name, type_) in schema.items()
            ],
        )
