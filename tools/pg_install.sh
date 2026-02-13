# Tutorial for building pgvector

git clone https://github.com/pgvector/pgvector.git 
cd pgvector

sudo apt update
sudo apt install build-essential libpq-dev postgresql-server-dev-all
# or, if you want a specific version (e.g., 15):
# sudo apt install postgresql-server-dev-15

# make
make PG_CONFIG=$(which pg_config)
sudo make install

# start pg service
sudo service postgresql start
sudo service postgresql status
# sudo service postgresql stop


# Create User
sudo -u postgres psql -c "CREATE USER asplos25 WITH PASSWORD '123456';"
sudo -u postgres psql -c "CREATE DATABASE database_asplos OWNER asplos25;"
sudo -u postgres psql database_asplos -c "CREATE EXTENSION vector;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE database_asplos TO asplos25;"
PGPASSWORD=123456 psql -h localhost -U asplos25 -d database_asplos -c "SELECT version();"



