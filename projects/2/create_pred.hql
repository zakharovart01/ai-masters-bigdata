CREATE TEMPORARY EXTERNAL TABLE IF NOT EXISTS hw2_pred(
id int,
preds Float)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES('separatorChar'='\t')
STORED AS TEXTFILE
LOCATION 'zakharovart01_hw2_pred';
