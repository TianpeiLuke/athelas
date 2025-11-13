from pathlib import Path

from sagemaker.pytorch import PyTorch

from sagemaker.network import NetworkConfig

from mods.mods_template import MODSTemplate
from sagemaker import Session, TrainingInput
from sagemaker.processing import ProcessingOutput, ProcessingInput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from secure_ai_sandbox_workflow_python_sdk.mims_model_registration.mims_model_registration_processing_step import (
    MimsModelRegistrationProcessingStep,
)

from secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_processor import (
    CradleDataLoader,
)
from secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_step import (
    CradleDataLoadingStep,
)

KMS_ENCRYPTION_KEY_PARAM = ParameterString(
    name="KMS_ENCRYPTION_KEY_PARAM", default_value="a"
)
VPC_SUBNET = ParameterString(name="VPC_SUBNET", default_value="b")
PIPELINE_EXECUTION_TEMP_DIR = ParameterString(
    name="EXECUTION_S3_PREFIX", default_value="a"
)

SECURITY_GROUP_ID = ParameterString(name="SECURITY_GROUP_ID", default_value="c")


@MODSTemplate(
    author="maxueyu", description="EU TSA Suspect Queue Model", version="1.1.0"
)
class EUTSASuspectQueueModel:
    def __init__(self, sagemaker_session=None):
        self.sagemaker_session = sagemaker_session or Session()
        self.load_training_data = self.create_training_data_loading_step()
        self.load_validation_data = self.create_validation_data_loading_step()
        self.load_calibration_data = self.create_calibration_data_loading_step()
        self.preprocessing_step_training = self.create_preprocessing_step_training()
        self.preprocessing_step_validation = self.create_preprocessing_step_validation()
        self.preprocessing_step_calibration = (
            self.create_preprocessing_step_calibration()
        )
        self.training_step = self.create_training_step()
        self.generic_rfuge_step = self.create_generic_rfuge_step()
        self.mims_packaging_step = self.create_mims_packaging_step()
        self.registration_step = self.create_registration_step()

    def generate_pipeline(self):
        return Pipeline(
            # Change this pipeline name if there is role based policy error
            name="EUTSAModel",
            parameters=[
                KMS_ENCRYPTION_KEY_PARAM,
                VPC_SUBNET,
                SECURITY_GROUP_ID,
                PIPELINE_EXECUTION_TEMP_DIR,
            ],
            sagemaker_session=self.sagemaker_session,
            steps=[
                self.load_training_data,
                self.load_validation_data,
                self.load_calibration_data,
                self.preprocessing_step_training,
                self.preprocessing_step_validation,
                self.preprocessing_step_calibration,
                self.training_step,
                self.generic_rfuge_step,
                self.mims_packaging_step,
                self.registration_step,
            ],
        )

    def create_training_data_loading_step(self):
        data_loading_step = CradleDataLoadingStep(
            step_name="Training_Data_Download",
            role=self.sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=self.sagemaker_session,
        )
        return data_loading_step

    def create_validation_data_loading_step(self):
        data_loading_step = CradleDataLoadingStep(
            step_name="Validation_Data_Download",
            role=self.sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=self.sagemaker_session,
        )
        return data_loading_step

    def create_calibration_data_loading_step(self):
        data_loading_step = CradleDataLoadingStep(
            step_name="Calibration_Data_Download",
            role=self.sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=self.sagemaker_session,
        )
        return data_loading_step

    def create_preprocessing_step_training(self):
        training_data_location = self.load_training_data.get_output_locations("DATA")

        sklearn_processor = SKLearnProcessor(
            framework_version="1.2-1",
            instance_type="ml.r5.24xlarge",
            instance_count=1,
            volume_size_in_gb=2048,
            role=self.sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=self.sagemaker_session,
            output_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            volume_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            network_config=NetworkConfig(
                enable_network_isolation=True,
                security_group_ids=[SECURITY_GROUP_ID],
                subnets=[VPC_SUBNET],
                encrypt_inter_container_traffic=True,
            ),
        )

        return ProcessingStep(
            name="Preprocessing_Train",
            processor=sklearn_processor,
            code=str(Path(__file__).parent / "scripts" / "preprocess_train.py"),
            inputs=[
                ProcessingInput(
                    source=training_data_location,
                    destination="/opt/ml/processing/input/training",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "config"),
                    destination="/opt/ml/processing/input/config",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "scripts"),
                    destination="/opt/ml/processing/input/config/scripts",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent.parent / "python_packages"),
                    destination="/opt/ml/processing/input/python_packages",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="output", source="/opt/ml/processing/output"
                ),
            ],
            depends_on=[
                self.load_training_data,
            ],
        )

    def create_preprocessing_step_validation(self):
        validation_data_location = self.load_validation_data.get_output_locations(
            "DATA"
        )

        sklearn_processor = SKLearnProcessor(
            framework_version="1.2-1",
            instance_type="ml.r5.24xlarge",
            instance_count=1,
            volume_size_in_gb=2048,
            role=self.sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=self.sagemaker_session,
            output_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            volume_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            network_config=NetworkConfig(
                enable_network_isolation=True,
                security_group_ids=[SECURITY_GROUP_ID],
                subnets=[VPC_SUBNET],
                encrypt_inter_container_traffic=True,
            ),
        )

        return ProcessingStep(
            name="Preprocessing_Vali",
            processor=sklearn_processor,
            code=str(Path(__file__).parent / "scripts" / "preprocess_vali.py"),
            inputs=[
                ProcessingInput(
                    source=validation_data_location,
                    destination="/opt/ml/processing/input/validation",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "config"),
                    destination="/opt/ml/processing/input/config",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "scripts"),
                    destination="/opt/ml/processing/input/config/scripts",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent.parent / "python_packages"),
                    destination="/opt/ml/processing/input/python_packages",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="output", source="/opt/ml/processing/output"
                ),
            ],
            depends_on=[
                self.load_validation_data,
            ],
        )

    def create_preprocessing_step_calibration(self):
        calibration_data_location = self.load_calibration_data.get_output_locations(
            "DATA"
        )

        sklearn_processor = SKLearnProcessor(
            framework_version="1.2-1",
            instance_type="ml.r5.24xlarge",
            instance_count=1,
            volume_size_in_gb=2048,
            role=self.sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=self.sagemaker_session,
            output_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            volume_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            network_config=NetworkConfig(
                enable_network_isolation=True,
                security_group_ids=[SECURITY_GROUP_ID],
                subnets=[VPC_SUBNET],
                encrypt_inter_container_traffic=True,
            ),
        )

        return ProcessingStep(
            name="Preprocessing_Cali",
            processor=sklearn_processor,
            code=str(Path(__file__).parent / "scripts" / "preprocess_cali.py"),
            inputs=[
                ProcessingInput(
                    source=calibration_data_location,
                    destination="/opt/ml/processing/input/calibration",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "config"),
                    destination="/opt/ml/processing/input/config",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "scripts"),
                    destination="/opt/ml/processing/input/config/scripts",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent.parent / "python_packages"),
                    destination="/opt/ml/processing/input/python_packages",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="output", source="/opt/ml/processing/output"
                ),
            ],
            depends_on=[
                self.load_calibration_data,
            ],
        )

    def create_training_step(self):
        import time

        seconds = int(time.time())
        use_small_batch = 0

        use_amp = 1
        use_moe = 0
        patience = 5

        # Change based on data
        num_seq = "1"
        model_name = "OrderFeature"
        batch_size = 800
        n_cat_features = 108
        n_num_features = 51
        n_embedding = 2403
        n_engineered_num_features = 413

        seq_len = 51
        hyperparameters = {
            "n_cat_features": n_cat_features,
            "n_num_features": n_num_features,
            "n_embedding": n_embedding,
            "n_engineered_num_features": n_engineered_num_features,
            "seq_len": seq_len,
            "dim_embedding_table": 64,
            "dim_attn_feedforward": 256,
            "num_heads": 1,
            "n_layers_order": 6,
            "n_layers_feature": 4,
            "num_seq": num_seq,
            "batch_size": batch_size,
            "max_epoch": 100,
            "seed": 0,
            "optim": "adam",
            "lr": 5e-05,
            "scheduler_maxlr": 0.0005,
            "beta1": 0.9,
            "beta2": 0.999,
            "emb_tbl_use_bias": 1,
            "dropout": 0.1,
            "use_small_batch": use_small_batch,
            "seconds": seconds,
            "use_time_seq": 1,
            "use_mlp": 0,
            "use_moe": use_moe,
            "use_amp": use_amp,
            "num_experts": 5,
            "modelname": model_name,
            "test_name": f"""isSmall_{use_small_batch}_OrderFeature""",
            "loss": "CrossEntropyLoss",
            "data_version": "v0",
            "load_model": 0,
            "model_path": "N/A",
            "steps_per_epoch": 2000,
            "patience": patience,
        }

        estimator = PyTorch(
            entry_point="train.py",
            source_dir="scripts",
            role=self.sagemaker_session.get_caller_identity_arn(),
            image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
            instance_count=1,
            instance_type="ml.g5.48xlarge",
            #             instance_type="ml.p3.16xlarge", volume_size=2048,
            hyperparameters=hyperparameters,
            max_run=86400 * 5,
            sagemaker_session=self.sagemaker_session,
            subnets=[VPC_SUBNET],
            security_group_ids=[SECURITY_GROUP_ID],
            enable_sagemaker_metrics=True,
            metric_definitions=[
                {
                    "Name": "average_training_loss",
                    "Regex": "average_training_loss=(.*?);",
                },
                {
                    "Name": "average_validation_loss",
                    "Regex": "average_validation_loss=(.*?);",
                },
            ],
            distribution={"torch_distributed": {"enabled": True}},
        )

        return TrainingStep(
            name="Training",
            estimator=estimator,
            inputs={
                "train": self.preprocessing_step_training.properties.ProcessingOutputConfig.Outputs[
                    "output"
                ].S3Output.S3Uri,
                "vali": self.preprocessing_step_validation.properties.ProcessingOutputConfig.Outputs[
                    "output"
                ].S3Output.S3Uri,
                "cali": self.preprocessing_step_calibration.properties.ProcessingOutputConfig.Outputs[
                    "output"
                ].S3Output.S3Uri,
            },
            depends_on=[
                self.preprocessing_step_training,
                self.preprocessing_step_validation,
                self.preprocessing_step_calibration,
            ],
        )

    def create_generic_rfuge_step(self):
        script_processor = ScriptProcessor(
            role=self.sagemaker_session.get_caller_identity_arn(),
            command=["Rscript"],
            image_uri="743349767511.dkr.ecr.us-east-1.amazonaws.com/r-temporary:latest",
            instance_type="ml.m5.4xlarge",
            instance_count=1,
            volume_size_in_gb=64,
            sagemaker_session=self.sagemaker_session,
            output_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            volume_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            network_config=NetworkConfig(
                enable_network_isolation=True,
                security_group_ids=[SECURITY_GROUP_ID],
                subnets=[VPC_SUBNET],
                encrypt_inter_container_traffic=True,
            ),
        )

        return ProcessingStep(
            name="generic_rfuge",
            processor=script_processor,
            code=str(Path(__file__).parent / "scripts" / "generic_rfuge.r"),
            inputs=[
                ProcessingInput(
                    source=self.training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/input/data",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent.parent / "r_packages"),
                    destination="/opt/ml/processing/input/r_packages",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="output", source="/opt/ml/processing/output"
                ),
            ],
            depends_on=[self.training_step],
        )

    def create_mims_packaging_step(self):
        data_output_path = "/opt/ml/processing/output"
        processor = SKLearnProcessor(
            framework_version="1.0-1",
            role=self.sagemaker_session.get_caller_identity_arn(),
            volume_size_in_gb=100,
            instance_type="ml.m5.4xlarge",
            instance_count=1,
            sagemaker_session=self.sagemaker_session,
            volume_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            network_config=NetworkConfig(
                enable_network_isolation=True,
                security_group_ids=[SECURITY_GROUP_ID],
                subnets=[VPC_SUBNET],
                encrypt_inter_container_traffic=True,
            ),
        )

        return ProcessingStep(
            name="AddInferenceDependencies",
            description="Add dependency files for inference",
            processor=processor,
            code=str(Path(__file__).parent / "scripts" / "mims_package.py"),
            inputs=[
                ProcessingInput(
                    source=self.training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/input/model",
                ),
                ProcessingInput(
                    source=self.generic_rfuge_step.properties.ProcessingOutputConfig.Outputs[
                        "output"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/bspline",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "inference_script"),
                    destination="/opt/ml/processing/input/inference_script",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "scripts"),
                    destination="/opt/ml/processing/input/inference_dependencies",
                ),
                ProcessingInput(
                    source=str(Path(__file__).parent / "config"),
                    destination="/opt/ml/processing/input/config",
                ),
            ],
            outputs=[ProcessingOutput(output_name="output", source=data_output_path)],
            depends_on=[self.generic_rfuge_step],
        )

    def create_registration_step(self):
        inputs = ProcessingInput(
            source=self.mims_packaging_step.properties.ProcessingOutputConfig.Outputs[
                "output"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/input/model",
        )
        return MimsModelRegistrationProcessingStep(
            step_name="MimsModelRegistrationProcessingStep",
            role=self.sagemaker_session.get_caller_identity_arn(),
            sagemaker_session=self.sagemaker_session,
            processing_input=[inputs],
            depends_on=[self.mims_packaging_step],
        )
