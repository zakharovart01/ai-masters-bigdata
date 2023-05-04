from datetime import datetime
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator 
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
import os

dsenv = "/opt/conda/envs/dsenv/bin/python"

base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'
train_path_in = '/datasets/amazon/all_reviews_5_core_train_extra_small_sentiment.json'
test_path_in = '/datasets/amazon/all_reviews_5_core_test_extra_small_features.json'
train_path_out = 'zakharovart01_train_out'
test_path_out = 'zakharovart01_test_out'
n_proj = '6'
prediction_path_out = 'zakharovart01_hw6_prediction'

with DAG(
    dag_id = "zakharovart01_dag",
    schedule_interval = None,
    start_date = datetime(2022, 5, 6),
    catchup = False
) as dag:
    
    feature_eng_train_task = SparkSubmitOperator(
          task_id = "feature_eng_train_task",
          application=f"{base_dir}/feature_eng_train.py",
          application_args = ["--path-in", train_path_in, "--path-out", train_path_out],
          spark_binary = "/usr/bin/spark-submit",
          env_vars={"PYSPARK_PYTHON": dsenv}
    )
    download_train_task = BashOperator(
          task_id = "download_train_task",
          bash_command = "hdfs dfs -getmerge {} {}/{}_local".format(train_path_out, base_dir, train_path_out)
    )
    train_task = BashOperator(
          task_id = "train_task",
          bash_command = f"{dsenv} {base_dir}/train.py --train-in  {base_dir}/{train_path_out}_local --sklearn-model-out {base_dir}/{n_proj}"
    )
    model_densor = FileSensor(
        task_id = "model_sensor",
        filepath = f"{base_dir}/{n_proj}.joblib",
        poke_interval = 30,
        timeout=600
    )
    feature_eng_test_task = SparkSubmitOperator(
          task_id = "feature_eng_test_task",
          application=f"{base_dir}/feature_eng_test.py",
          application_args = ["--path-in", test_path_in, "--path-out", test_path_out],
          spark_binary = "/usr/bin/spark-submit",
          env_vars={"PYSPARK_PYTHON": dsenv}
    )
    predict_task = SparkSubmitOperator(
        task_id = "predict_task",
        application = f"{base_dir}/predict.py",
        application_args = ["--test-in", test_path_out, "--pred-out", prediction_path_out, "--sklearn-model-in", f"{base_dir}/{n_proj}.joblib"],
        spark_binary = "/usr/bin/spark-submit",
        env_vars={"PYSPARK_PYTHON": dsenv}
    )


    feature_eng_train_task >> download_train_task >> train_task >> model_densor >> feature_eng_test_task >> predict_task