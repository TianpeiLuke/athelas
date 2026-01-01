import datetime
import logging
import os

import digital_polars_utils

logger = logging.getLogger(__name__)

ABS_PATH = os.path.dirname(os.path.realpath(__file__))

try:
    from secure_ai_sandbox_python_lib.session import Session

    from com.amazon.secureaisandboxproxyservice.models.createcradledataloadjobrequest import (
        CreateCradleDataLoadJobRequest,
    )
    from com.amazon.secureaisandboxproxyservice.models.datasourcesspecification import (
        DataSourcesSpecification,
    )
    from com.amazon.secureaisandboxproxyservice.models.mdsdatasourceproperties import (
        MdsDataSourceProperties,
    )
    from com.amazon.secureaisandboxproxyservice.models.andesdatasourceproperties import (
        AndesDataSourceProperties,
    )
    from com.amazon.secureaisandboxproxyservice.models.transformspecification import (
        TransformSpecification,
    )
    from com.amazon.secureaisandboxproxyservice.models.outputspecification import (
        OutputSpecification,
    )
    from com.amazon.secureaisandboxproxyservice.models.cradlejobspecification import (
        CradleJobSpecification,
    )
    from com.amazon.secureaisandboxproxyservice.models.edxdatasourceproperties import (
        EdxDataSourceProperties,
    )

    from com.amazon.secureaisandboxproxyservice.models.jobsplitoptions import (
        JobSplitOptions,
    )

    from com.amazon.secureaisandboxproxyservice.models.field import Field

    from com.amazon.secureaisandboxproxyservice.models.datasource import DataSource
    from secure_ai_sandbox_python_lib.utils import coral_utils

    logger.warning(f"Secure AI Sandbox tools are loaded successfully")

    import uuid
    import os
    import polars as pl
    from glob import glob

except:
    logger.warning(f"Secure AI Sandbox tools ARE NOT loaded")


class SAISEDXLoadJob:
    """
    Loads data via SAIS
    """

    def __init__(
        self,
        region,
        start_date,
        end_date,
        split_job=False,
        **kwargs,
    ):
        self.start_date = start_date
        self.end_date = end_date

        self.split_job = split_job

        self.s3_path = None
        self.cradle_data_load_job = None
        self.cradle_data_loader = None
        self.region = region

    def submit(self):

        mds_vars = {
            "objectId",
            "customerId",
            "orderDate",
            "transactionDate",
            "gls",
            "daysSinceFirstCompletedOrder",
            "marketplaceCountryCode",
            "asins",
            "finalDecision",
            "customerName",
            "billingAddressName",
            "emailAddress",
            "paymentAccountHolderName",
            "emailDomain",
            "orderTotalAmountUSD",
            "creditCardNegtableHit",
            "ipAddressNegtableHit",
            "currentUbidNegtableHit",
            "numNewSigninStates7Days",
            "geoBillCountryCcCountryCodeEqual",
            "evMaxLinkScoreOfAllCategory",
            "billingAddress",
            "billingCity",
            "billingState",
            "billingZipCode",
            "billingCountryCode",
        }

        tabular_vars = set()

        with open("features/DigitalModelNA.txt") as file:
            tabular_vars.update(line.strip() for line in file)

        with open("features/DigitalModelEU.txt") as file:
            tabular_vars = tabular_vars.intersection({line.strip() for line in file})

        with open("features/DigitalModelJP.txt") as file:
            tabular_vars = tabular_vars.intersection({line.strip() for line in file})

        if self.region == "NA":
            org_id = 1
        elif self.region == "EU":
            org_id = 2
        elif self.region == "FE":
            org_id = 9
        else:
            raise ValueError("Invalid org id")

        mds_vars.update(tabular_vars)

        request = CreateCradleDataLoadJobRequest(
            data_sources=DataSourcesSpecification(
                start_date=self.start_date.isoformat(),  # data start date
                end_date=self.end_date.isoformat(),  # data end date
                data_sources=[  # data sources a list of data source properties
                    DataSource(
                        data_source_name="D_CUSTOMERS",
                        data_source_type="ANDES",
                        andes_data_source_properties=AndesDataSourceProperties(
                            provider="booker",
                            table_name="D_CUSTOMERS",
                            andes3_enabled=True,  # Set to true if your Andes table has Andes 3.0 enable with latest version.
                        ),
                    ),
                    DataSource(
                        data_source_name="RAW_MDS",  # data source name, it should be uniq across the list of data source. this name should be used as table name when you write the SQL
                        data_source_type="MDS",  # data source type, it can be 'MDS/ANDES/EDX', you need setup the properties according to this type
                        mds_data_source_properties=MdsDataSourceProperties(  #
                            service_name="FORTRESS",
                            org_id=org_id,
                            region=self.region,
                            output_schema=[
                                Field(field_name=col, field_type="STRING")
                                for col in mds_vars
                            ],
                            use_hourly_edx_data_set=False,
                        ),
                    ),
                ],
            ),
            transform_specification=TransformSpecification(  # transformSQL should refer the above data source name to query the data
                transform_sql=f"""
                    WITH features AS (
                        SELECT
                            RAW_MDS.*,
                            D_CUSTOMERS.status AS status,
                            ROW_NUMBER() OVER (PARTITION BY RAW_MDS.objectId ORDER BY RAW_MDS.transactionDate) AS dedup
                        FROM RAW_MDS
                            INNER JOIN D_CUSTOMERS ON RAW_MDS.customerId = D_CUSTOMERS.customer_id
                        WHERE ABS(daysSinceFirstCompletedOrder) < 1e-12
                    )
                    SELECT *
                    FROM features
                    WHERE dedup = 1
                        AND ((status = 'N' AND RAND() < 0.5) OR status IN ('F', 'I'))
                    """,
                job_split_options=JobSplitOptions(
                    split_job=self.split_job,  # You can enable job split option by changing this function to True, but you need provide merge_sql. INPUT will the all the data after split executes, you can write extra logic in SQL, e.g. using group by for statistics or dedup.
                    days_per_split=30,
                    merge_sql="""
                        select * from INPUT
                    """,
                ),
            ),
            output_specification=OutputSpecification(
                output_schema=list(mds_vars) + ["status"],
                output_path=self.s3_path,  #
                output_format="PARQUET",  # output format can be CSV, UNESCAPED_TSV, JSON, ION, PARQUET. CSV is the default format if you don't specify it
                output_save_mode="ERRORIFEXISTS",  # output save mode can setup to support different case, it can be OVERWRITE, ERRORIFEXISTS, APPEND, IGNORE. In default it's ERRORIFEXISTS. ",
                output_file_count=0,  # output file count can be set to reduce or increase final the number of files. Too many output files will cause S3 throttling failure; Too few output will encounter performance issues. current setting is 30 per day, you can provide the overrides for your overrides.
                keep_dot_in_output_schema=False,  # When set to true, the output file header will contain normal the '.'. Otherwise when set to False, the output file header will replace every '.' with '__DOT__'.,
                include_header_in_s3_output=True,
            ),
            cradle_job_specification=CradleJobSpecification(
                cluster_type="LARGE",
                cradle_account="BRP-ML-Payment-Generate-Data",
                extra_spark_job_arguments="",  # you can customize the spark job driver memory if you need by vending parameters here
                job_retry_count=0,  # job retry count in case of failure, in default Cradle will retry once if it fails. you can customize retry times.
            ),
        )

        cradle_data_load_job = self.cradle_data_loader.create_cradle_data_load_job(
            request
        )

        return cradle_data_load_job

    def run(self, **kwargs):

        sandbox_session = Session(
            session_folder="/home/ec2-user/SageMaker/cradle_output"
        )
        self.cradle_data_loader = sandbox_session.resource("CradleDataLoader")

        self.s3_path = f"s3://{sandbox_session.my_owned_s3_bucket_name()}/mds_download_output/{uuid.uuid4()}"
        print(self.s3_path)

        logger.info(f"output path: {self.s3_path}")

        self.cradle_data_load_job = self.submit()

    def download(self, output_path=None):
        self.cradle_data_loader.wait_for_done(
            self.cradle_data_load_job, sleep_interval=60
        )

        digital_polars_utils.download_parquet_s3(
            os.path.join(self.s3_path, "*"),
            output_path
            or f"{datetime.datetime.utcnow().isoformat(timespec='seconds')}_{self.region}.parquet".replace(
                ":", "-"
            ),
        )


if __name__ == "__main__":
    jobs = []
    for region in ["NA", "EU", "FE"]:
        job = SAISEDXLoadJob(
            region,
            datetime.datetime(2025, 2, 15),
            datetime.datetime(2025, 5, 15),
        )
        job.run()
        jobs.append(job)

    for job in jobs:
        job.download()
