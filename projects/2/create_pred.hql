CREATE TABLE hw2_pred(
    id INT,
    pred float
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION 'zakharovart01_hw2_pred';
