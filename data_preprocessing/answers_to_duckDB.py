import duckdb
con = duckdb.connect("dataset/cert42.duckdb") 

#only insiders.csv will go in as scenario based TP user logs are already present in the og log files
file_path = "dataset/answers/insiders.csv"


sql_query = """CREATE TABLE insiders AS 
SELECT * FROM read_csv_auto(?) 
WHERE dataset = '4.2';"""


con.execute(sql_query, [file_path])
res = con.execute("SELECT * from insiders LIMIT 5").df() 
print(res) 