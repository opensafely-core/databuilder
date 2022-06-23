import os
import secrets
import time
from pathlib import Path

import sqlalchemy
import sqlalchemy.exc
from sqlalchemy.dialects import registry
from sqlalchemy.orm import sessionmaker

from .util import iter_flatten

MSSQL_SETUP_DIR = Path(__file__).parents[1].absolute() / "support/mssql"


# Register our modified PyHive SQLAlchemy dialect
registry.register(
    "spark.pyhive.opensafely",
    "databuilder.query_engines.spark_dialect",
    "SparkDialect",
)


class DbDetails:
    def __init__(
        self,
        protocol,
        driver,
        host_from_container,
        port_from_container,
        host_from_host,
        port_from_host,
        username="",
        password="",
        db_name="",
        query=None,
        temp_db=None,
    ):
        self.protocol = protocol
        self.driver = driver
        self.host_from_container = host_from_container
        self.port_from_container = port_from_container
        self.host_from_host = host_from_host
        self.port_from_host = port_from_host
        self.password = password
        self.username = username
        self.db_name = db_name
        self.query = query
        self.temp_db = temp_db

    def container_url(self):
        return self._url(self.host_from_container, self.port_from_container)

    def host_url(self):
        return self._url(self.host_from_host, self.port_from_host)

    def engine(self, dialect=None, **kwargs):
        url = self._url(self.host_from_host, self.port_from_host, include_driver=True)
        engine_url = sqlalchemy.engine.make_url(url)
        # We always want the "future" API
        engine = sqlalchemy.create_engine(engine_url, future=True, **kwargs)
        return engine

    def _url(self, host, port, include_driver=False):
        assert self.username and self.password
        auth = f"{self.username}:{self.password}@"

        if include_driver:
            protocol = f"{self.protocol}+{self.driver}"
        else:
            protocol = self.protocol
        url = f"{protocol}://{auth}{host}:{port}/{self.db_name}"
        # only used for databricks atm
        if self.query:  # pragma: no cover
            url += "?" + "&".join(f"{k}={v}" for k, v in self.query.items())
        return url

    def setup(self, *input_data, metadata=None):
        """
        Accepts SQLAlchemy ORM objects (which may be arbitrarily nested within lists and
        tuples), creates the necessary tables and inserts them into the database
        """
        input_data = list(iter_flatten(input_data))

        engine = self.engine()
        Session = sessionmaker()
        Session.configure(bind=engine)
        session = Session()

        if metadata:
            pass
        elif input_data:
            metadata = input_data[0].metadata
        else:
            assert False, "No source of metadata"
        assert all(item.metadata is metadata for item in input_data)

        metadata.drop_all(engine)
        metadata.create_all(engine)
        session.bulk_save_objects(input_data)

        if self.temp_db is not None:
            session.execute(
                sqlalchemy.text(f"CREATE DATABASE IF NOT EXISTS {self.temp_db}")
            )
        session.commit()


def wait_for_database(database, timeout=10):
    engine = database.engine()

    start = time.time()
    limit = start + timeout
    while True:
        try:
            with engine.connect() as connection:
                connection.execute(sqlalchemy.text("SELECT 'hello'"))
            break
        except (
            sqlalchemy.exc.OperationalError,
            ConnectionRefusedError,
            ConnectionResetError,
            BrokenPipeError,
        ) as e:  # pragma: no cover
            if time.time() >= limit:
                raise Exception(
                    f"Failed to connect to database after {timeout} seconds: "
                    f"{engine.url}"
                ) from e
            time.sleep(1)


def make_mssql_database(containers):
    password = "Your_password123!"

    container_name = "databuilder-mssql"
    mssql_port = 1433

    if not containers.is_running(container_name):  # pragma: no cover
        run_mssql(container_name, containers, password, mssql_port)

    container_ip = containers.get_container_ip(container_name)
    host_mssql_port = containers.get_mapped_port_for_host(container_name, mssql_port)

    return DbDetails(
        protocol="mssql",
        driver="pymssql",
        host_from_container=container_ip,
        port_from_container=mssql_port,
        host_from_host="localhost",
        port_from_host=host_mssql_port,
        username="SA",
        password=password,
        db_name="test",
    )


def run_mssql(container_name, containers, password, mssql_port):  # pragma: no cover
    containers.run_bg(
        name=container_name,
        image="mcr.microsoft.com/mssql/server:2017-CU25-ubuntu-16.04",
        volumes={
            MSSQL_SETUP_DIR: {"bind": "/mssql", "mode": "ro"},
        },
        # Choose an arbitrary free port to publish the MSSQL port on
        ports={mssql_port: None},
        environment={
            "SA_PASSWORD": password,
            "ACCEPT_EULA": "Y",
            "MSSQL_TCP_PORT": str(mssql_port),
        },
        entrypoint="/mssql/entrypoint.sh",
        command="/opt/mssql/bin/sqlservr",
    )


def make_spark_database(containers):
    if "DATABRICKS_URL" in os.environ:  # pragma: no cover
        return make_databricks_database()
    else:
        return make_spark_container_database(containers)


def make_spark_container_database(containers):
    container_name = "databuilder-spark"
    # This is the default anyway, but better to be explicit
    spark_port = 10001

    if not containers.is_running(container_name):  # pragma: no cover
        containers.run_bg(
            name=container_name,
            # Nothing special about this particular version other than that
            # it's the latest as of the time of writing
            image="docker.io/bitnami/spark:3.1.2-debian-10-r126",
            entrypoint="/bin/bash",
            command=[
                # To speak SQL to our Spark database we need to start a thing
                # called Hive which speaks a protocol called Thrift. Command
                # below cribbed from:
                # https://github.com/bitnami/bitnami-docker-spark/issues/32#issuecomment-820668226
                "/opt/bitnami/spark/bin/spark-submit",
                "--class",
                "org.apache.spark.sql.hive.thriftserver.HiveThriftServer2",
                # By default Hive tries to set cookies on the client which
                # breaks both the client libraries I've tried so we disable
                # that here
                "--hiveconf",
                "hive.server2.thrift.http.cookie.auth.enabled=false",
                # Use the HTTP (as opposed to binary) protocol
                "--hiveconf",
                "hive.server2.transport.mode=http",
                "--hiveconf",
                f"hive.server2.thrift.http.port={spark_port}",
            ],
            # As described below, there's a directory permissions issue when
            # trying to run Hive using this container. It's possible to chmod
            # the relevant directory, but given that this is a test environment
            # it's easier just to run as root. See:
            # https://github.com/bitnami/bitnami-docker-spark/issues/32
            user="root",
            # Supplying a host port of None tells Docker to choose an arbitrary
            # free port
            ports={spark_port: None},
        )

    container_ip = containers.get_container_ip(container_name)
    host_spark_port = containers.get_mapped_port_for_host(container_name, spark_port)

    return DbDetails(
        protocol="spark",
        driver="pyhive+opensafely",
        host_from_container=container_ip,
        port_from_container=spark_port,
        host_from_host="localhost",
        port_from_host=host_spark_port,
        # These are arbitrary but we need to supply _some_ values here
        username="foo",
        password="bar",
        temp_db="tempdb",
    )


# This is not used by regular units tests, only by manual invocation, hence why
# coverage is skipped
def make_databricks_database():  # pragma: no cover
    url = os.environ["DATABRICKS_URL"]
    url = sqlalchemy.engine.make_url(url)

    return DbDetails(
        protocol="spark",
        driver="pyhive+opensafely",
        host_from_container=url.host,
        port_from_container=url.port or 443,
        host_from_host=url.host,
        port_from_host=url.port or 443,
        username=url.username,
        password=url.password,
        db_name=url.database,
        query=url.query,
        temp_db="tempdb",
    )


class InMemorySQLiteDatabase(DbDetails):
    def __init__(self):
        db_name = secrets.token_hex(8)
        super().__init__(
            db_name=db_name,
            protocol="sqlite",
            driver="pysqlite",
            host_from_container=None,
            port_from_container=None,
            host_from_host=None,
            port_from_host=None,
        )
        self._engine = None

    def engine(self, dialect=None, **kwargs):
        # We need to hold a reference to the engine for the lifetime of this database to stop the contents of the
        # database from being garbage-collected.
        if not self._engine:
            self._engine = super().engine(dialect, **kwargs)
        return self._engine

    def _url(self, host, port, include_driver=False):
        if include_driver:
            protocol = f"{self.protocol}+{self.driver}"
        else:
            protocol = self.protocol
        # https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#uri-connections
        # https://sqlite.org/inmemorydb.html
        return f"{protocol}:///file:{self.db_name}?mode=memory&cache=shared&uri=true"
