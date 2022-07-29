import contextlib
import logging
import secrets
import time

import sqlalchemy

log = logging.getLogger(__name__)


def write_query_to_table(table, query):
    """
    Write the results of `query` into `table` (both SQLAlchemy objects)
    """
    # This is a bit of a hack but works OK. We want to be able to take
    # an arbitrary select query and generate the SQL:
    #
    #   SELECT * INTO some_temporary_table FROM (some_select_query) AS some_alias
    #
    # which is the MSSQL-specific syntax for writing the results of a
    # query directly into a temporary table. We can trick SQLAlchemy
    # into generating this for us by giving it a literal column named
    # "* INTO table_name".
    into_table = sqlalchemy.literal_column(f"* INTO {table.name}")
    return sqlalchemy.select(into_table).select_from(query.alias())


@contextlib.contextmanager
def fetch_results_in_batches(
    engine,
    queries,
    temp_table_prefix=None,
    key_column="patient_id",
    **batch_fetch_config,
):
    """
    Context manager which takes a list of queries and executes them, returning
    an iterator over the results of the final query.
    Behind the scenes this writes the results to a temporary table and then
    downloads the results in batches, retrying after any errors. This allows us
    to workaround flakiness in our connection to the database.
    Args:
        engine: sqlalchemy.Engine instance
        queries: list of sqlalchemy queries, with the final being a Select
        temp_table_prefix: prefix to use when generating temporary table names
            use a leading `#` character for session-scoped temporary tables,
            use the "double dot" `DatabaseName..table_prefx` syntax to store
            tables in another database
        key_column: name of a unique integer column in the results, used for paging
        batch_size: how many results to fetch in each batch
        max_retries: how many *sequential* failures to retry after
        sleep: time in seconds to sleep between retries
        reconnect_on_error: whether to close and reopen database connection
            after each error; this is robust against more forms of failure, but
            can't be used with session-scoped temporary tables for obvious
            reasons
    """
    preparatory_queries = queries[:-1]
    select_query = queries[-1]
    assert isinstance(select_query, sqlalchemy.sql.expression.Select)

    with ReconnectableConnection(engine) as connection:
        # The temporary table name we use contains a random string which
        # means this temporary table will only be accessible within
        # the lifetime of this ReconnectableConnection,
        # but since it is not a true MSSQL temporary table it will
        # persist across MSSQL sessions in case of reconnection

        table_name = f"[{temp_table_prefix}_{secrets.token_hex(8)}]"
        table = make_table_with_key(table_name, key_column)
        assert_temporary_tables_writable(connection, temp_table_prefix)
        for n, query in enumerate(preparatory_queries):
            log.info(f"Running query {n}/{len(queries)}")
            connection.execute(query)
        # Run the write to temporary table within an explicit transaction
        # so we can't end up in a state where the table exists but is
        # half-populated. We have to commit the existing implicit
        # transaction before we can start the new one.
        connection.commit()
        with connection.begin():
            log.info(f"Running final query and writing results to '{table_name}'")
            connection.execute(write_query_to_table(table, select_query))
            log.info(f"Creating '{key_column}' index on '{table_name}'")
            connection.execute(create_index_for_table(table))
            connection.commit()

        # Here we pass back an iterator over the temporary table as the value
        # of the context manager
        yield fetch_table_in_batches(
            connection, table, key_column, **batch_fetch_config
        )

        # If all goes well we clean up the temporary table; if there was an
        # error then we'll leave it in place for next time
        connection.execute(sqlalchemy.schema.DropTable(table, if_exists=True))
        connection.commit()


def fetch_table_in_batches(
    connection,
    table,
    key_column,
    batch_size=32000,
    max_retries=2,
    sleep=0.5,
    reconnect_on_error=False,
):
    """
    Returns an iterator over all the rows in a table by querying it in batches,
    retrying automatically on errors.
    Args:
        connection: SQLAlchemy Connection object, or ReconnectableConnection
            (see below)
        table: SQLAlchemy Table object
        key_column: name of a unique integer column in the results, used for
            paging (note that this will need an index on it to avoid terrible
            performance)
        batch_size: how many results to fetch in each batch
        max_retries: how many *sequential* failures to retry after
        sleep: time in seconds to sleep between retries
        reconnect_on_error: whether to close and reopen database connection
            after each error; this is robust against more forms of failure (e.g
            the underlying TCP connection being closed), but can't be used with
            session-scoped temporary tables for obvious reasons. If this option
            is supplied then `connection` must be a ReconnectableConnection
            instance
    """
    if reconnect_on_error:
        msg = "Connection must be a ReconnectableConnection if `reconnect_on_error` is used"
        assert hasattr(connection, "reconnect"), msg

    key = sqlalchemy.Column(key_column)
    retries = 0
    batch_count = 0
    total_rows = 0
    min_key = None

    while True:
        query = (
            sqlalchemy.select("*").select_from(table).order_by(key).limit(batch_size)
        )
        if min_key is not None:
            query = query.where(key > min_key)

        log.info(f"Fetching batch {batch_count}")
        try:
            results = connection.execute(query)
            retries = 0
        except sqlalchemy.exc.OperationalError as e:
            retries += 1
            if retries > max_retries:
                raise
            else:
                log.info(f"Error: {e!r}\nAttempting retry {retries}/{max_retries}")
                if reconnect_on_error:
                    log.info("Closing database connection")
                    connection.reconnect()
                time.sleep(sleep)
                continue

        row_count = 0
        for row in results:
            row_count += 1
            yield row
            min_key = row[key_column]

        total_rows += row_count
        batch_count += 1
        log.info(f"Total rows fetched: {total_rows}")

        if row_count < batch_size:
            log.info("Batch fetch complete")
            break


class ReconnectableConnection:
    """
    Wraps a `sqlalchemy.engine.Connection` to add a `reconnect()` method which
    closes the underlying connection and then re-opens it when next used. This
    simplifies code elsewhere by allowing it to act as if it's dealing with a
    single connection object.
    This doesn't attempt to implement the whole `Connection` interface, just
    enough of it to work for our purposes. And obviously if you try to do
    something silly like reconnect in the middle of a transaction then
    behaviour is undefined.
    """

    _conn = None

    def __init__(self, engine, **connect_kwargs):
        self.engine = engine
        self.connect_kwargs = connect_kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._conn is not None:
            self._conn.close()

    @property
    def connection(self):
        if self._conn is None:
            self._conn = self.engine.connect(**self.connect_kwargs)
        return self._conn

    @property
    def dialect(self):
        return self.connection.dialect

    def execute(self, *args, **kwargs):
        return self.connection.execute(*args, **kwargs)

    def commit(self, *args, **kwargs):
        return self.connection.commit(*args, **kwargs)

    def begin(self, *args, **kwargs):
        return self.connection.begin(*args, **kwargs)

    def reconnect(self):
        """
        Close the underlying connection and reconnect next time we need it
        """
        if self._conn is None:
            return
        # Remove the connection from the pool so that calling `close()` will
        # actually close it rather than just returning it to the pool
        self._conn.detach()
        self._conn.close()
        self._conn = None


def table_exists(connection, table):
    # We don't always have sufficient permissions to check this in the "proper"
    # way so we have to try reading from it and catch the error
    try:
        results = connection.execute(sqlalchemy.select(1).select_from(table).limit(1))
        # Consume the results
        list(results)
        return True
    except sqlalchemy.exc.DBAPIError as e:
        # This obviously isn't great, but we don't get a more specific error
        # class out of SQLAlchemy. Trying to use error numbers ties us to
        # driver implementation details so in fact looking for this specific
        # string turns out to be the most robust thing we can do here.
        if "Invalid object name" in str(e):
            return False
        else:
            raise  # pragma: no cover


def assert_temporary_tables_writable(connection, temp_table_prefix):
    """
    Check that we can write temporary tables before we start potentially very
    long running queries. The most surefire way to do this is to try to write
    to a test table and see what happens.
    """
    random_suffix = secrets.token_hex(8)
    test_table_name = f"{temp_table_prefix}_test_{random_suffix}"
    test_table = make_table_with_key(test_table_name, "id")
    test_query = sqlalchemy.select(sqlalchemy.literal(1).label("foo"))
    try:
        connection.execute(write_query_to_table(test_table, test_query))
        connection.execute(sqlalchemy.schema.DropTable(test_table))
    except sqlalchemy.exc.DBAPIError as e:
        message = f"Unable to write temporary table with prefix '{temp_table_prefix}'"
        raise RuntimeError(message) from e


def make_table_with_key(table_name, key_column):
    return sqlalchemy.Table(
        table_name,
        sqlalchemy.MetaData(),
        sqlalchemy.Column(key_column, sqlalchemy.Integer, index=True),
        # Because these table names may be fully qualified (i.e.
        # "DatabaseName.SchemaName.TableName") we don't want SQLAlchemy to
        # quote them for us. Because these names are fixed in the code and not
        # user supplied we aren't worried about anything breaking here.
        quote=False,
    )


def create_index_for_table(table):
    indexes = list(table.indexes)
    assert len(indexes) == 1
    return sqlalchemy.schema.CreateIndex(indexes[0])
