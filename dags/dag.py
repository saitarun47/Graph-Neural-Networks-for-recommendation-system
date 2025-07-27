from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from gnn_pipeline import load_data , load_and_process , build_graph , split_and_sample , store_in_neo4j, train_gnn_model , evaluate_gnn_model


default_args={
        "owner":'airflow',
        "start_date": days_ago(1),
        "email":['tanuj00047@gmail.com'],
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=1)
}

## create the dag

dag = DAG(
    'gnn_dag',
    default_args=default_args,
    description='etl code',
    schedule_interval='@once'
)

## python operator to run the dag

run_etl = PythonOperator(
    task_id='gnn_etl',
    python_callable=load_data,
    dag=dag
)

run_etl2 = PythonOperator(
    task_id='gnn_etl2',
    python_callable=load_and_process,
    provide_context = True,
    dag=dag
)

run_etl3 = PythonOperator(
    task_id='gnn_etl3',
    python_callable=build_graph,
    provide_context = True,
    dag=dag
)

run_etl4 = PythonOperator(
    task_id='gnn_etl4',
    python_callable=split_and_sample,
    provide_context = True,
    dag=dag
)

store_data = PythonOperator(
    task_id='store_data',
    python_callable=store_in_neo4j,
    provide_context = True,
    dag=dag
)

training = PythonOperator(
    task_id='gnn_train',
    python_callable=train_gnn_model,
    provide_context = True,
    dag=dag
)


eval = PythonOperator(
    task_id='eval',
    python_callable=evaluate_gnn_model,
    provide_context = True,
    dag=dag
)
run_etl >> run_etl2 >> run_etl3 >> run_etl4 >> store_data >> training >> eval
