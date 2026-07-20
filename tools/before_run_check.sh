# check pg DB
sudo service postgresql status
PGPASSWORD=123456 psql -h localhost -U asplos25 -d database_asplos -c "SELECT version();"
# check models
python ./tools/list_hf_model.py

