import datetime
import os
import re
import uuid
from collections import defaultdict


import boto3
import redshift_connector
import polars as pl
from scipy.stats import ks_2samp
from pyarrow.parquet import ParquetFile


from secure_ai_sandbox_python_lib.session import Session
from com.amazon.secureaisandboxproxyservice.models.createcradlemdsdataloadjobrequest import (
    CreateCradleMDSDataLoadJobRequest,
)

TRANSACTION_DATE_REGEX = r"^[A-Z][a-z]{2}\s[A-Z][a-z]{2}\s[0-3][0-9]\s[0-2][0-9]:[0-5][0-9]:[0-5][0-9]\sUTC\s\d{4}$"
ORDER_DATE_REGEX = r"^\d+$"


def query(query: str, cluster: str = "1"):
    # Pick one cluster and collect the following information form onboarded clusters information wiki:
    # https://w.amazon.com/bin/view/CMLS/SecureAISandbox/Guides/RedShiftFromNotebook#HOnboardedRedShiftClusters
    # This example picks redshift-cluster-1 which is the test cluster and should be accessible from all the onboarded SAIS aws account.

    CLUSTER_INFOS = {
        "1": {
            "cluster_id": "trmsml-rs-01",
            "db_name": "trmsmlprod",
            "endpoint": "vpce-svc-0d214190d5b3f7c10.us-east-1.vpce.amazonaws.com",
            "arn": "arn:aws:iam::359048081192:role/SecureAISandbox_Role_Andes3",
        },
        "2": {
            "cluster_id": "trmsml-rs-02",
            "db_name": "trmsmlprod2",
            "endpoint": "vpce-svc-0e3212a137fbe2ae3.us-east-1.vpce.amazonaws.com",
            "arn": "arn:aws:iam::359048081192:role/SecureAISandbox_Role_Andes3",
        },
        "adhoc": {
            "cluster_id": "trmsdw-rs-adhoc",
            "db_name": "trmsopsadhoc",
            "endpoint": "vpce-svc-00fc97a9cb28c8be0.us-east-1.vpce.amazonaws.com",
            "arn": "arn:aws:iam::084474670698:role/Secure_AI_Sandbox_Andes30",
        },
    }

    cluster_info = CLUSTER_INFOS[cluster]

    # Use Session to automatically configure some environment variables
    session = Session(session_folder=".")
    sts = boto3.client("sts")
    creds = sts.assume_role(
        RoleArn=cluster_info["arn"],
        RoleSessionName="sais_session",
        Tags=[
            {"Key": "currentOwnerAlias", "Value": session.owner_alias()},
        ],
        TransitiveTagKeys=["currentOwnerAlias"],
    )["Credentials"]

    # Connect
    redshift_boto3 = boto3.client(
        "redshift",
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )

    redshift_login = redshift_boto3.get_cluster_credentials_with_iam(
        ClusterIdentifier=cluster_info["cluster_id"],
        DurationSeconds=3600,
        DbName=cluster_info["db_name"],
    )

    connection = redshift_connector.connect(
        database=cluster_info["db_name"],
        host=cluster_info["endpoint"],
        port=8192,
        user=redshift_login["DbUser"],
        password=redshift_login["DbPassword"],
    )

    with connection:
        df = pl.read_database(
            query, connection=connection, infer_schema_length=None
        ).cast({pl.Decimal: pl.Float64})

    return df


def my_bucket():
    bucket_name = Session(
        session_folder="/home/ec2-user/SageMaker/"
    ).my_owned_s3_bucket_name()
    return f"s3://{bucket_name}"


def upload_s3(filepath, s3_path=None):

    s3_path = s3_path or os.path.basename(filepath)

    assert "s3:" not in s3_path

    s3_loader = Session(session_folder="/home/ec2-user/SageMaker/").resource(
        "MyOwnS3BucketDataLoader"
    )
    s3_loader.upload_file(filepath, s3_path)


def ks_test(df, target):

    df = df.lazy()

    ks_stats = []

    for col in (
        df.select(pl.selectors.numeric().exclude(target)).collect_schema().names()
    ):

        df_col = df.select(col, target).drop_nulls().collect()

        df_good = df_col.filter(pl.col(target) == 0)
        df_bad = df_col.filter(pl.col(target) == 1)

        if df_good.is_empty() or df_bad.is_empty():
            continue

        ks_stat = ks_2samp(df_good[col], df_bad[col])

        ks_stats.append(
            {"name": col, "ks_stat": ks_stat.statistic, "ks_p": ks_stat.pvalue}
        )

    return pl.DataFrame(ks_stats).sort("ks_stat", descending=True)


def corr_with_target(
    df: pl.DataFrame | pl.LazyFrame, target: str = "IS_FRD", method="spearman"
) -> pl.DataFrame:
    return (
        df.lazy()
        .select(pl.corr(pl.selectors.numeric(), pl.col(target), method=method))
        .unpivot(value_name=method)
        .filter(
            pl.col(method).is_not_nan(),
            ~pl.col("variable")
            .str.to_lowercase()
            .str.contains_any(["audit", "decision", "ruleid", "randomnumber"]),
        )
        .sort(pl.col(method).abs(), descending=True)
        .collect()
    )


def evaluate_rule(
    df: pl.DataFrame | pl.LazyFrame,
    rule: str,
    return_orders: bool = True,
    true: pl.Expr = pl.col("IS_FRD") == 1,
) -> pl.LazyFrame:
    
    df = df.lazy()

    usd_name = (
        "orderTotalAmountUSD"
        if "orderTotalAmountUSD" in df.columns
        else "ordertotalamountusd"
    )

    rule = (
        re.sub(r"([^0-9])\.([^0-9])", r"\1__DOT__\2", rule)
        .replace("$", "")
        .replace("&&", "AND")
        .replace("||", "OR")
    )

    pred = pl.sql_expr(rule)

    df_perf = df.lazy().select(
        fn=(~pred & true).sum(),
        tp=(pred & true).sum(),
        fp=(pred & ~true).sum(),
        bad_usd=pl.col(usd_name).filter(pred & true).sum(),
        good_usd=pl.col(usd_name).filter(pred & ~true).sum(),
        yield_usd=pl.col(usd_name)
        .filter(pred)
        .sum()
        .truediv((pred & true).sum() + (pred & ~true).sum())
        .round(2),
    )

    if return_orders:
        return df.filter(pred), df_perf

    return df_perf


def infer_schema(filename: str):

    DTYPE_HIERARCHY = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.Boolean]
    
    def _infer_column_type(s: pl.Series):
        for dtype in DTYPE_HIERARCHY:
            try:
                s.cast(dtype)
                return dtype
            except pl.exceptions.InvalidOperationError:
                pass

        return pl.String
    
    possible_schemas = defaultdict(set)

    with ParquetFile(filename) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=10_000):

            df = pl.from_arrow(batch)

            for s in df:
                dtype = _infer_column_type(s)
                possible_schemas[s.name].add(dtype)
                
    return {
        col: next(
            dtype
            for dtype in [pl.String] + DTYPE_HIERARCHY[::-1]
            if dtype in dtypes
        )
        for col, dtypes in possible_schemas.items()
    }


def download_parquet_s3(s3_glob, output_file, infer=True):

    if "s3:" not in s3_glob:
        s3_glob = os.path.join(my_bucket(), s3_glob)

    if not infer:
        (
            pl.scan_parquet(s3_glob)
            .cast({pl.Decimal: pl.Float64})
            .sink_parquet(output_file)
        )
        return

    tmp_file = f"/tmp/{uuid.uuid4()}.parquet"

    try:
        (
            pl.scan_parquet(s3_glob)
            .cast({pl.Decimal: pl.Float64})
            .sink_parquet(tmp_file)
        )

        schema = infer_schema(tmp_file)
        transforms = {}
            
        if "IS_FRD" in schema:
            schema["IS_FRD"] = pl.Int8
            
        if "orderDate" in schema:
            schema["orderDate"] = pl.String
            transforms["orderDate"] = (
                pl.when(pl.col("orderDate").str.contains(ORDER_DATE_REGEX))
                .then(pl.col("orderDate"))
                .otherwise(None)
                .str.to_datetime("%s")
            )

        if "transactionDate" in schema:
            schema["transactionDate"] = pl.String
            transforms["transactionDate"] = (
                pl.when(pl.col("transactionDate").str.contains(TRANSACTION_DATE_REGEX))
                .then(pl.col("transactionDate"))
                .otherwise(None)
                .str.to_datetime("%a %b %d %H:%M:%S %Z %Y")
            )

        (
            pl.scan_parquet(tmp_file)
            .cast(schema)
            .with_columns(**transforms)
            .sink_parquet(output_file)
        )

    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


class MDSDataJob:
    def __init__(
        self,
        *,
        features: list[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        sql_filter: str = "",
        tag: str = "",
        s3_rel_path: str = "",
        split_days: int = None,
        service: str = "FORTRESS",
        region: str = "EU",
        cluster_type: str = "LARGE",
        dedup: str = "KEEP_FIRST",
    ):
        assert len(features) == len(set(features))
        assert region in ("EU", "NA", "FE")

        self.features = features
        self.start_date = start_date
        self.end_date = end_date
        self.sql_filter = sql_filter
        self.tag = tag.replace(" ", "_").replace(":", "-")
        self.dedup = dedup

        self.data_loader = None
        self.data_load_job = None
        self.session = Session(session_folder="/home/ec2-user/SageMaker/mds_output")
        self.s3_path = (
            f"s3://{self.session.my_owned_s3_bucket_name()}/{s3_rel_path}"
            if s3_rel_path
            else f"s3://{self.session.my_owned_s3_bucket_name()}/mds_download_output/{uuid.uuid4()}"
        )
        self.split_days = split_days
        self.service = service
        self.region = region
        self.cluster_type = cluster_type

    def run(self):
        self.data_loader = self.session.resource("MDSDataLoader")

        self.data_load_job = self.data_loader.create_mds_data_load_job(
            CreateCradleMDSDataLoadJobRequest(
                # Your service name and org id can be found https://unified-ml-catalog.ctps.amazon.dev/list-mds-datasets
                service_name=self.service,
                org_id={"EU": "2", "NA": "1", "FE": "9"}[self.region],
                # org_id=9001,
                # region should be NA/EU/FE/CN
                region=self.region,
                # Start date
                start_date=self.start_date.strftime("%Y-%m-%dT%H:%M:%S"),
                # End Date
                end_date=self.end_date.strftime("%Y-%m-%dT%H:%M:%S"),
                # output path currently only support s3, format should be 's3://bucket/key_prefix'
                output_path=self.s3_path,
                # output schema is a list of field names,  use this tool to query your MDS data set signature: https://unified-ml-catalog.ctps.amazon.dev/query-mds-signature, and then select variables you like.
                output_schema=self.features,
                # output format can be CSV, UNESCAPED_TSV, JSON, ION, PARQUET. CSV is the default format if you don't specify it
                output_format="PARQUET",
                # cluster type could be STANDARD,SMALL,MEDIUM,LARGE, make sure your cradle account has the type you selected. for more type information  https://w.amazon.com/bin/view/BDT/Products/Cradle/GettingStarted/EstimateClusterSizes/
                cluster_type=self.cluster_type,
                # dedup type can be 'NONE'/'KEEP_FIRST'/'KEEP_LAST'
                dedup_type=self.dedup,
                # Each aws account have different account permission, check here to check the cradle account you can use: https://secure-ai-sandbox.ctps.amazon.dev/account
                cradle_account="BRP-ML-Payment-Generate-Data",
                # Filter condition's item should be in output_schema so we can do the filtering correctly
                filter_conditions=self.sql_filter,
                # You can just pass the SQL by yourself without using generated SQL by MDSDataLoader if you are familiar with SQL. it's suggested to use this so you can make it more efficient.
                # Example: 'select 'customerId' from INPUT where object!=null'
                customized_sql="",
                # Auto Tag pulls the tags from fraud tags  Andes table: https://hoot.corp.amazon.com/providers/26b27bde-3847-49c6-a07c-0289c17d9c33/tables/fraud-tags-na
                auto_tag=True,
                # Add more tag fields here: https://hoot.corp.amazon.com/providers/26b27bde-3847-49c6-a07c-0289c17d9c33/tables/fraud-tags-na
                auto_tag_schema=["IS_FRD", "HAS_CB", "IS_QUEUED"],
                # You can provide a EDX Arn as tag file input, EDX ARN could be manifest, manifest prefix, or manifest_range.
                # The tag file format can be text/csv, text/tsv, application/x-amzn-unescaped-tsv.
                # Please make sure you provide tag_file_schema parameters if your EDX data format is not 'objectId,__TAG__'
                # You can use EDXDataLoader to upload your tag file into your team owned edx data set, check the doc here: https://w.amazon.com/bin/view/CMLS/SecureAISandbox/Guides/EdxAccess/
                # Some example tag file input:
                # tag_file=f"arn:amazon:edx:iad::manifest/{EDX_PROVIDER}/{EDX_SUBJECT}/{EDX_DATASET}/["datasetkey"]",
                # tag_file=f"arn:amazon:edx:iad::manifest_prefix/{EDX_PROVIDER}/{EDX_SUBJECT}/{EDX_DATASET}/["prefix"]",
                # tag_file=f"arn:amazon:edx:iad::manifest_range/{EDX_PROVIDER}/{EDX_SUBJECT}/{EDX_DATASET}/?range_start=["start_key"]&range_end=["end_key"]&completeness_check_type=throw", # In this case, Cradle will wait for the complete data
                # tag_file=f"arn:amazon:edx:iad::manifest_range/{EDX_PROVIDER}/{EDX_SUBJECT}/{EDX_DATASET}/?range_start=["start_key"]&range_end=["end_key"]&completeness_check_type=no_check",  # In this case, Cradle will start as long as there is at least one manifest returned.
                # Please also make sure your cradle account has the permission to access this EDX data set
                tag_file=None,
                # in default tag file schema is ["objectId", "__TAG__"], if it's different, please provide one, make sure there is no field name conflicts between tag files schema and output_schema.
                tag_file_schema=None,
                # Users can decide if the job can be splitted, set it False means you want to do the splitting.
                # WARNING: to avoid the create unnecessary multiple split jobs , we set this parameter as True. please change it to False if you want to split the job per days.
                no_split=True if self.split_days is None else False,
                # Users can define each split contains how many days data, in default it's 15 days. you can adjust it as you want.
                days_per_split=self.split_days,
                # Users can customize how many output files that writes to S3 depending the data size so we can improve the performance. in default, it will be caculated as 30 output file per day. if you pull 5 days, it will be output as 150 files.
                output_file_count=20,
                keep_dot_in_output_schema=False,
                # cradle_job_provider="SAIS",
                job_retry_count=4,
            )
        )

    def download_from_s3(self, output_path=None):
        if self.data_loader is not None:
            self.data_loader.wait_for_done(self.data_load_job, sleep_interval=60)

        download_parquet_s3(
            os.path.join(self.s3_path, "*"),
            output_path
            or f"/home/ec2-user/SageMaker/SecureAISandbox-CodeCommitRepository-kkrowpma-us-east-1/data/{self.tag}.parquet",
        )
